import sys
import torch
sys.path.insert(0, './')
import numpy as np
from libs.data.preprocessor import get_preprocessor
from libs.body_model import BodyModel
from libs.models.utils import HWC3
import joblib as pkl
import cv2

meta = {
    'gen_width': 512,
    'gen_height': 512,
}

poses = np.load('./datasets/DanceDB/20120731_StefanosTheodorou/Stefanos_1os_antrikos_karsilamas_C3D_poses.npz', allow_pickle=True)
shape = np.load("./datasets/DanceDB/20120731_StefanosTheodorou/shape.npz", allow_pickle=True)

bm = BodyModel("./deps/body_models/smplh/neutral/model.npz")
t_pose = bm()

poses = torch.from_numpy(poses["poses"]).float()

index = range(0, poses.shape[0], 1000)

pose = bm(pose_body=poses[index, 3:66], root_orient=poses[index, :3])

data = {
    "vertices": pose.v,
}

preprocessor = get_preprocessor(meta)
data = preprocessor(data)

images = data["rasterized_segments"]
for idx in range(images.shape[0]):
    cv2.imwrite("./datasets/DanceDB_img/{}.jpg".format(idx), images[idx])