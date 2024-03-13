import os
import torch
import random
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.transforms import (
    rotation_6d_to_matrix
)
from libs.smplx import SMPLHLayer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from libs.datamodule.utils import mld_collate

class HumanML3DDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config.datasets["HumanML3D"]

        self.joints_num = 22

        self.val = None
        self.test = None
        self.train = None

        self.mean = np.load(os.path.join(self.data_dir, 'Mean.npy'))
        self.std = np.load(os.path.join(self.data_dir, 'Std.npy'))

        self.mean[:] = 0.
        self.std[:] = 1.

        self.normalize = config.normalize
        self.text_zero_padding = torch.load(os.path.join(self.data_dir, 'text_embeddings/-1.pt'))
        self.bodymodel = SMPLHLayer(model_path=config.smpl_path)
        self.mean_tensor = torch.from_numpy(self.mean).float()
        self.std_tensor = torch.from_numpy(self.std).float()

    def setup(self, stage: str):
        self.val = HumanML3D(
            self.data_dir, 
            data_fields='val', 
            mean=self.mean if self.normalize else None,
            std=self.std if self.normalize else None
        )
        self.train = HumanML3D(
            self.data_dir, 
            data_fields='train_val',
            mean=self.mean if self.normalize else None,
            std=self.std if self.normalize else None
        )

    def test_dataloader(self):
        if self.test is None:
            self.test = HumanML3D(
                self.data_dir,
                data_fields='test',
                mean=self.mean if self.normalize else None,
                std=self.std if self.normalize else None
            )
        return 
        

    def train_dataloader(self):
        return DataLoader(
            self.train, 
            batch_size=self.config.train.batch_size,
            shuffle=True,
            collate_fn=mld_collate,
            num_workers=self.config.train.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, 
            batch_size=self.config.train.batch_size,
            shuffle=False,
            collate_fn=mld_collate,
            num_workers=self.config.train.num_workers,
            pin_memory=True
        )
    
    def inv_transform(self, data):
        return data * self.std_tensor.to(data.device) + self.mean_tensor.to(data.device)
    
    def recover_motion(self, data):
        # 22 * 9
        if data.shape[-1] == 198:
            transl = data[..., :3]
            body_pose = rotation_6d_to_matrix(data[..., 72:].reshape(-1, 22, 6))
            global_orient = rotation_6d_to_matrix(data[..., 66:72].reshape(-1, 6))
        else:
            assert NotImplementedError

        body = self.bodymodel(body_pose=body_pose, global_orient=global_orient, transl=transl)
        return body


class HumanML3D(Dataset):
    """HumanML3D: a pytorch loader for motion text dataset"""

    def __init__(self, dataset_dir, data_fields, mean=None, std=None, text_embedders=[]) -> None:
        super().__init__()
        assert os.path.exists(dataset_dir), dataset_dir

        # hyperparameters
        min_motion_len = 30
        self.unit_length = 4
        self.joints_num = 22
        self.normalize = (mean is not None)

        self.mean = mean
        self.std = std

        self.motion_dir = os.path.join(dataset_dir, 'joints_new')
        self.text_dir = os.path.join(dataset_dir, 'texts')
        self.text_embedding_dir = os.path.join(dataset_dir, 'text_embeddings')

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
                motion = np.load(os.path.join(self.motion_dir, name + '.npy'))
                motion = motion[:, 22*3:]

                if (len(motion)) < min_motion_len or (len(motion) >= 256):
                    continue
                text_data = []
                flag = False
                with cs.open(os.path.join(self.text_dir, name + '.txt')) as f:
                    text_embedding = torch.load(os.path.join(self.text_embedding_dir, name + '.pt'))
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
        self.max_length = max(self.length_arr)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = item
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

        # padding zero
        # if m_length < self.max_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_length - m_length, motion.shape[1]))
        #                              ], axis=0, dtype=np.float32)
        

        return (
            motion, 
            m_length, 
            caption,
            token
        )
