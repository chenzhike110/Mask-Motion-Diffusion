import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

def YZswitch(root_rot, root_trans):
    """
    switch Y axis Z axis
    """
    batch_size = root_rot.shape[0]
    transform = torch.tensor(
        [[1., 0., 0., 0.],
         [0., 0., 1., 0.],
         [0., 1., 0., 0.],
         [0., 0., 0., 1.]]
    ).unsqueeze(0).repeat(batch_size, 1, 1).to(root_rot.device)
    init_state = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(root_rot.device)
    init_state[:, :-1, :-1] = axis_angle_to_matrix(root_rot)
    init_state[:, :-1, -1] = root_trans
    
    transformed = torch.bmm(transform, init_state)
    root_rot = matrix_to_axis_angle(transformed[:, :-1, :-1])
    root_trans = transformed[:, :-1, -1]
    return root_rot, root_trans
    