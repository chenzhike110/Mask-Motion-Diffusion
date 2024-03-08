import os
import torch
import trimesh
import numpy as np
from libs.interaction.utils import to_Tensor, create_mat, mat2rt
from libs.render.utils import YZswitch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, matrix_to_axis_angle

########################################
## Interaction Target Sample class
########################################
class Interactions():
    def __init__(
        self, 
        humanpose,
        weights=None,
        device="cpu",
    ):
        self.smpl_offset = torch.Tensor([-0.0018, -0.2233,  0.0282]).to(device)
        if torch.is_tensor(humanpose):
            self.humanpose = humanpose.to(device)
        elif type(humanpose) == str:
            humanpose = torch.from_numpy(np.load(humanpose)).to(device)
            self.obj_rot = humanpose[:, :3]
            smpl_trans = (humanpose[:, 6:9] - humanpose[:, 3:6]) + self.smpl_offset
            # print(smpl_trans[0])
            smpl_rot = humanpose[:, 9:12]
            self.humanpose = humanpose[:, 6:]
            self.humanpose[:, 3:6] = matrix_to_axis_angle(torch.bmm(torch.linalg.inv(axis_angle_to_matrix(self.obj_rot)), axis_angle_to_matrix(smpl_rot)))
            self.humanpose[:, :3] = torch.bmm(torch.linalg.inv(axis_angle_to_matrix(self.obj_rot)), smpl_trans.unsqueeze(-1)).squeeze(-1).to(device)
            # print(humanpose[0, :3])
            
            # rot, trans = YZswitch(rot, trans)
            # self.humanpose[:, :3] = trans
            # self.humanpose[:, 3:6] = rot
            
            
        print("load distribution of ", self.humanpose.shape)
        if weights is None:
            self.weights = torch.ones(self.humanpose.shape[0]) / self.humanpose.shape[0]
        else:
            self.weights = weights

        self.device = device

        assert self.weights.shape[0] == self.humanpose.shape[0]

    def process_data(
        self, 
        human_npz, 
        object_npz, 
        object_mesh, 
        dump_dir, 
        dump_imgs=False,
        thres=0.001
    ):
        import cv2
        import igl
        import pickle
        from tqdm import tqdm
        from libs.body_model import BodyModel
        from libs.render.behave import Pyt3DWrapper
        
        bm = BodyModel('deps/body_models/smplh/neutral/model.npz').to(self.device)
        human_part_segm = pickle.load(open("deps/body_models/parts_segm/smplh/parts_segm.pkl", 'rb'), encoding='latin1')['segm']

        if dump_imgs:
            os.makedirs(os.path.join(dump_dir, 'imgs'), exist_ok=True)
            image_size = 960
            w, h = image_size, int(image_size * 0.75)
            pyt3d_wrapper = Pyt3DWrapper(image_size=(w, h), device=self.device)

        contact_indexes = []
        bar = tqdm(range(human_npz["poses"].shape[0]))
        datas = []
        for i in bar:
            smpl = bm(
                root_orient=to_Tensor(human_npz["poses"][i, None, :3], self.device),
                pose_body=to_Tensor(human_npz["poses"][i, None, 3:66], self.device),
                pose_hand=to_Tensor(human_npz["poses"][i, None, 66:156], self.device),
                trans=to_Tensor(human_npz["trans"][i, None, :], self.device),
                return_dict=True
            )
            # print(human_npz["trans"][i, None, :])
            # print(human_npz["poses"][i, None, :3])
           

            smpl = {k:v.squeeze(0).cpu().numpy() for k, v in smpl.items()}

            angle, trans = object_npz['angles'][i, None, :], object_npz['trans'][i, None, :]
            obj_trans = create_mat(angle, trans, 'rot_vec')
            
            # print(matrix_to_euler_angles(axis_angle_to_matrix(torch.from_numpy(angle)), 'XYZ'))
            
            rot_obj = object_mesh.copy().apply_transform(obj_trans)

            dist, face_index, vertices = igl.signed_distance(rot_obj.vertices, smpl['v'], smpl['f'], return_normals=False)
            contact = dist<thres
            if np.any(contact):
                if np.any(human_part_segm[face_index[contact]]<1):
                    contact_indexes.append(i)
                    # smpl_trans = create_mat(human_npz["poses"][i, None, :3], human_npz["trans"][i, None, :], 'rot_vec')
                    # smpl2obj = np.linalg.inv(obj_trans) @ smpl_trans
                    
                    # print(angle, trans)
                    # print(human_npz["poses"][i, None, :3], human_npz["trans"][i, None, :])
                    # exit(0)

                    # rot, _ = mat2rt(smpl2obj)
                    # print(rot, trans)
                    # exit(0)
                    # human_npz["trans"][i, None, :] = trans
                    # human_npz["poses"][i, None, :3] = rot
                    datas.append(np.concatenate([
                        object_npz['angles'][i, :], object_npz['trans'][i, :],
                        human_npz["trans"][i, :], human_npz["poses"][i, :]
                    ], axis=-1))

                    if dump_imgs:
                        rot_obj = {'v': rot_obj.vertices, 'f': rot_obj.faces}
                        fit_meshes_local = [smpl, rot_obj]
                        rend = pyt3d_wrapper.render_meshes(fit_meshes_local)
                        rend = (rend*255).astype(np.uint8)
                        cv2.imwrite(os.path.join(dump_dir, 'imgs', f'{i}.jpg'), rend)
                        
            bar.set_postfix({'contact_num': len(contact_indexes)})

        contact_indexes = np.array(contact_indexes)

        datas = np.array(datas)
        
        np.save(
            os.path.join(dump_dir, 'distribution.npy'), 
            datas
        )
    
    def sample(self, num, objs):
        pose_id = torch.multinomial(self.weights,
                                    num_samples=num,
                                    replacement=True)
        smpl_poses = self.object_to_smpl(
            self.humanpose[pose_id].reshape(pose_id.shape[0], *self.humanpose.shape[1:]),
            objs
        )
        return smpl_poses
    
    def object_to_smpl(self, smpls, objs):
        objs = objs.to(self.device)
        smpls = smpls.to(self.device)
        
        smpl_trans = torch.eye(4).unsqueeze(0).repeat(smpls.shape[0], 1, 1).to(self.device)
        smpl_trans[..., :-1, :-1] = axis_angle_to_matrix(smpls[:, 3:6])
        smpl_trans[..., :-1, -1] = smpls[..., :3]
        
        smpl_trans = torch.bmm(objs, smpl_trans)
        
        root_rot = matrix_to_axis_angle(smpl_trans[:, :-1, :-1]).float()
        root_trans = smpl_trans[:,:-1,-1].float()

        root_trans -= self.smpl_offset
        return root_trans, root_rot, smpls[6:]
