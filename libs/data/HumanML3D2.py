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

class HumanML3D(Dataset):
    """HumanML3D: a pytorch loader for motion text dataset"""

    def __init__(self, dataset_dir, data_fields, normalize=False, text_encoder='clip') -> None:
        super().__init__()
        assert os.path.exists(dataset_dir), dataset_dir

        # hyperparameters
        min_motion_len = 30
        self.unit_length = 4
        self.joints_num = 22
        self.max_length = 19
        self.normalize = normalize

        self.motion_dir = os.path.join(dataset_dir, 'joints_smpl_pose6d')
        self.text_dir = os.path.join(dataset_dir, 'texts')
        self.mean = np.load(os.path.join(dataset_dir, 'smpl_Mean.npy'))
        self.std = np.load(os.path.join(dataset_dir, 'smpl_Std_new.npy'))
        # self.mean = np.zeros(4+3*(self.joints_num-1))
        # self.std = np.ones(4+3*(self.joints_num-1))
        # self.mean[:4] = mean[:4]
        # self.std[:4] = std[:4]
        
        self.mean_tensor = torch.from_numpy(self.mean).float()
        self.std_tensor = torch.from_numpy(self.std).float()

        self.ds = {}

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
        for name in id_list:
            try:
                motion = np.load(os.path.join(self.motion_dir, name + '.npz'), allow_pickle=True)
                motion = np.concatenate((motion['trans'],motion['root_orient']), axis=-1, dtype=np.float32)
                # trans = np.load(os.path.join(self.trans_dir, name + '.npy'))
                # vel x,y + height z + aixs angle rotation * joint_num
                # motion = np.concatenate((trans[:, :4], motion['poses'][:-1, :63]), axis=-1, dtype=np.float32)

                if (len(motion)) < min_motion_len or (len(motion) >= 201):
                    continue
                text_data = []
                flag = False
                with cs.open(os.path.join(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
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

    def recover_motion(self, data, rotation='6d', normalized=False):
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
        
        return r_pos, r_rot_quat, pose_body

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, _ = text_data['caption'], text_data['tokens']

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

        # padding zero
        # if m_length < self.padding_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.padding_length - m_length, motion.shape[1]))
        #                              ], axis=0, dtype=np.float32)
        
        # pose_body, pose_root, length, text
        pose_body = motion[:, 3:]
        pose_root = motion[:, :3]

        return (
            pose_body, 
            pose_root, 
            m_length, 
            caption
        )
