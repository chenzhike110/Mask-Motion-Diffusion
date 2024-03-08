import sys
sys.path.insert(0, './')
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_axis_angle
from libs.body_model import BodyModel

body_model = BodyModel("deps/body_models/smplh/neutral/model.npz")

root_oritation = matrix_to_axis_angle(euler_angles_to_matrix(torch.Tensor([[torch.pi, 0, 0]]), 'XYZ'))

smpl1 = body_model()
print(smpl1.Jtr[0, 0])

smpl2 = body_model(root_orient=root_oritation)
print(smpl2.Jtr[0, 0])
