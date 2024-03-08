import torch
import numpy as np
from scipy.spatial.transform import Rotation

def to_Tensor(array, device):
    return torch.from_numpy(array).float().to(device)

def create_mat(rot, transl, rot_type='matrix'):
    if rot_type =='rot_vec':
        rot = Rotation.from_rotvec(rot).as_matrix()
    elif rot_type != 'matrix':
        rot = Rotation.from_euler(rot_type, rot).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = transl
    return mat.astype(np.float32)

def mat2rt(mat, rot_type='rot_vec'):
    if rot_type =='rot_vec':
        rot = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    elif rot_type == 'quat':
        rot = Rotation.from_euler(rot_type, rot).as_quat()
    transl = mat[:3, 3]
    return rot, transl