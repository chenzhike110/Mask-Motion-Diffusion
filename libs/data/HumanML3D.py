import os
import torch
import random
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_axis_angle
)
from libs.models.transform import qinv, qrot
from libs.body_model import BodyModel
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from libs.data.utils import mld_collate

class HumanML3DDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config.DATASETS["HumanML3D"]

        self.joints_num = 22

        self.mean = np.load(os.path.join(self.data_dir, 'smpl_Mean.npy'))
        self.std = np.load(os.path.join(self.data_dir, 'smpl_Std_new.npy'))
        # self.mean = np.zeros(4+3*(self.joints_num-1))
        # self.std = np.ones(4+3*(self.joints_num-1))
        # self.mean[:4] = mean[:4]
        # self.std[:4] = std[:4]
        self.text_zero_padding = np.load(os.path.join(self.data_dir, 'text_embeddings/-1.npy'))
        self.bodymodel = BodyModel(config.SMPL_PATH)
        self.mean_tensor = torch.from_numpy(self.mean).float()
        self.std_tensor = torch.from_numpy(self.std).float()

    def setup(self, stage: str):
        self.val = HumanML3D(self.data_dir, data_fields='val')
        self.train = HumanML3D(self.data_dir, data_fields='train_val')

    def train_dataloader(self):
        return DataLoader(
            self.train, 
            batch_size=self.config.TRAIN.BATCH_SIZE,
            shuffle=True,
            collate_fn=mld_collate,
            num_workers=self.config.TRAIN.NUM_WORKERS,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, 
            batch_size=self.config.TRAIN.BATCH_SIZE,
            shuffle=False,
            collate_fn=mld_collate,
            num_workers=self.config.TRAIN.NUM_WORKERS,
            pin_memory=True
        )
    
    def inv_transform(self, data):
        return data * self.std_tensor.to(data.device) + self.mean_tensor.to(data.device)
    
    def recover_root_rot_pos(self, data):
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        '''Add Y-axis rotation to root position'''
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data[..., 3]
        return r_rot_quat, r_pos

    def recover_motion(self, data, rotation='6d', normalized=False, local_only=True):
        # data (batch, seq, 4+21*6/21*3)
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        if normalized:
            data = self.inv_transform(data)
        
        r_pos = r_rot_quat = None
        # recover body pose
        if rotation == 'axis':
            if data.shape[-1] == (self.joints_num - 1) * 3 + 4:
                r_rot_quat, r_pos = self.recover_root_rot_pos(data)
                r_rot_quat =  quaternion_to_axis_angle(r_rot_quat.view(-1, 4))
                r_pos = r_pos.view(-1, 3)
                pose_body = data[..., 4:].view(-1, (self.joints_num - 1) * 3)
            elif data.shape[-1] == (self.joints_num - 1) * 3:
                pose_body = data.view(-1, (self.joints_num - 1) * 3)
            elif data.shape[-1] == self.joints_num * 3:
                pose_body = data.view(-1, self.joints_num * 3)[:, 3:]
                r_rot_quat = data.view(-1, self.joints_num * 3)[:, :3]
            else:
                raise NotImplementedError
        elif rotation == '6d':
            if data.shape[-1] == (self.joints_num - 1) * 6 + 4:
                r_rot_quat, r_pos = self.recover_root_rot_pos(data)
                r_rot_quat =  quaternion_to_axis_angle(r_rot_quat.view(-1, 4))
                r_pos = r_pos.view(-1, 3)
                pose_body = data[..., 4:].reshape(-1, 6)
                pose_body = matrix_to_axis_angle(rotation_6d_to_matrix(pose_body)).view(-1, (self.joints_num - 1) * 3)
            elif data.shape[-1] == (self.joints_num - 1) * 6:
                pose_body = data.reshape(-1, 6)
                pose_body = matrix_to_axis_angle(rotation_6d_to_matrix(pose_body)).view(-1, (self.joints_num - 1) * 3)
            elif data.shape[-1] == self.joints_num * 6:
                pose_body = data[..., 6:].reshape(-1, 6)
                r_rot_quat = data[..., :6]
                pose_body = matrix_to_axis_angle(rotation_6d_to_matrix(pose_body)).view(-1, (self.joints_num - 1) * 3)
                r_rot_quat = matrix_to_axis_angle(rotation_6d_to_matrix(r_rot_quat)).view(-1, 3)
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
        
        if local_only:
            r_pos = r_rot_quat = None

        bm = self.bodymodel(
            pose_body=pose_body, 
            root_orient=r_rot_quat,
            trans=r_pos
        )
        
        return bm

class HumanML3D(Dataset):
    """HumanML3D: a pytorch loader for motion text dataset"""

    def __init__(self, dataset_dir, data_fields, mean=None, std=None, text_embedders=[]) -> None:
        super().__init__()
        assert os.path.exists(dataset_dir), dataset_dir

        # hyperparameters
<<<<<<< HEAD
        min_motion_len = 30
        self.unit_length = 4
        self.joints_num = 22
        self.max_length = 19
        self.normalize = (mean is not None)

        self.mean = mean
        self.std = std

        self.motion_dir = os.path.join(dataset_dir, 'joints_smpl_pose6d')
=======
        min_motion_len = 19
        self.unit_length = 4
        self.joints_num = 22
        self.max_length = 19
        self.padding_length = 199
        self.normalize = normalize

        self.motion_dir = os.path.join(dataset_dir, 'joints_smpl_processed')
>>>>>>> 483f71c25ad588786ffd71f1d7bec3df7dc04878
        self.text_dir = os.path.join(dataset_dir, 'texts')
        self.text_embedding_dir = os.path.join(dataset_dir, 'text_embeddings')

        self.ds = {}

<<<<<<< HEAD
=======
        self.root_mean = np.zeros(6, dtype=np.float32)
        self.root_std = np.zeros(6, dtype=np.float32)
        self.root_mean[:2] = self.mean[:2] 
        self.root_mean[2:] = self.mean[5:9]
        self.root_std[:2] = self.std[:2] 
        self.root_std[2:] = self.std[5:9]

>>>>>>> 483f71c25ad588786ffd71f1d7bec3df7dc04878
        if len(data_fields) == 0:
            return
        
        data_dict = {}
        id_list = []
        with cs.open(os.path.join(dataset_dir, data_fields+'.txt'), 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(os.path.join(self.motion_dir, name + '.npz'), allow_pickle=True)
<<<<<<< HEAD
                motion = np.concatenate((motion['trans'],motion['root_orient']), axis=-1, dtype=np.float32)
                # trans = np.load(os.path.join(self.trans_dir, name + '.npy'))
                # vel x,y + height z + aixs angle rotation * joint_num
                # motion = np.concatenate((trans[:, :4], motion['poses'][:-1, :63]), axis=-1, dtype=np.float32)

                if (len(motion)) < min_motion_len or (len(motion) >= 201):
=======
                trans = motion['trans']
                vel = trans[1:, :] - trans[:-1, :]
                # vel x,y + height z + aixs angle rotation * joint_num
                motion = np.concatenate((vel[:, :-1], trans[:-1, 2:], motion['poses'][:-1, :66]), axis=-1, dtype=np.float32)

                if (len(motion)) < min_motion_len or (len(motion) >= 200):
>>>>>>> 483f71c25ad588786ffd71f1d7bec3df7dc04878
                    continue
                text_data = []
                flag = False
                with cs.open(os.path.join(self.text_dir, name + '.txt')) as f:
                    text_embedding = np.load(os.path.join(self.text_embedding_dir, name + '.npy'))
                    for idx, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = text_embedding[idx].reshape(1, -1)
                        text_dict['tokens'] = caption
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        # assert length <= self.padding_length
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, token = text_data['caption'], text_data['tokens']

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        if self.normalize:
            motion = (motion - self.mean) / self.std
        else:
            motion[:, :6] = (motion[:, :6] - self.root_mean) / self.root_std

        # padding zero
        # if m_length < self.padding_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.padding_length - m_length, motion.shape[1]))
        #                              ], axis=0, dtype=np.float32)
        
        # pose_body, pose_root, length, text
<<<<<<< HEAD
        pose_body = motion[:, 3:]
        pose_root = motion[:, :3]

        return (
            pose_body, 
            pose_root, 
            m_length, 
            caption,
            token
        )
=======
        pose_body = motion[:, 6: 6 + (self.joints_num - 1) * 3]
        pose_root = motion[:, :6]

        return {"pose_body":pose_body, "pose_root":pose_root, "length":m_length, "text":caption}
>>>>>>> 483f71c25ad588786ffd71f1d7bec3df7dc04878
