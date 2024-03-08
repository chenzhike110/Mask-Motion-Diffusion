import sys
sys.path.insert(0, './')
import torch
import joblib
import numpy as np
from libs.body_model import BodyModel
from pytorch3d.transforms import quaternion_to_axis_angle

data = joblib.load('/data2/czk/PMM-HumanML3D/sample_data/amass_run_isaac.pkl')
motion_name = '0-ACCAD_Female1Running_c3d_C5 - walk to run_poses'

motion = data[motion_name]
body_model = BodyModel('./deps/body_models/smplh/neutral/model.npz')

root_trans = motion['root_trans_offset']
root_quat = torch.from_numpy(motion['pose_quat_global'])
pose_aa = torch.from_numpy(motion['pose_aa'])

root_aa = quaternion_to_axis_angle(root_quat)

smpls = body_model(
    pose_body=pose_aa, 
    root_orient=root_aa,
    trans=root_trans
)

np.save(motion_name+'.npy', smpls.v[..., [0,2,1]].numpy())