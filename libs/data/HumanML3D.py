import os
import torch
import random
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils.data import Dataset

class HumanML3D(Dataset):
    """HumanML3D: a pytorch loader for motion text dataset"""

    def __init__(self, dataset_dir, data_fields, normalize=False) -> None:
        super().__init__()
        assert os.path.exists(dataset_dir), dataset_dir

        # hyperparameters
        min_motion_len = 40
        self.unit_length = 4
        self.joints_num = 22
        self.max_length = 20
        self.padding_length = 199
        self.normalize = normalize

        self.motion_dir = os.path.join(dataset_dir, 'new_joint_vecs')
        self.text_dir = os.path.join(dataset_dir, 'texts')
        self.mean = np.load(os.path.join(dataset_dir, 'Mean.npy'))
        self.std = np.load(os.path.join(dataset_dir, 'Std.npy'))
        self.ds = {}
        
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
                motion = np.load(os.path.join(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
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
        assert length <= self.padding_length
        self.pointer = np.searchsorted(self.length_arr, length)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

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
        if m_length < self.padding_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.padding_length - m_length, motion.shape[1]))
                                     ], axis=0, dtype=np.float32)
        
        # pose_body, pose_root, length, text
        pose_body = motion[:, 4 + (self.joints_num - 1) * 3: 4 + (self.joints_num - 1) * 9]
        if not self.normalize:
            pose_root = (motion[:, :4] - self.mean[:4]) / self.std[:4]
        else:
            pose_root = motion[:, :4]

        return {"pose_body":pose_body, "pose_root":pose_root, "length":m_length, "text":caption, 'mean':self.mean, 'std':self.std}
