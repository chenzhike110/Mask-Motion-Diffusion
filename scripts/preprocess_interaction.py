import sys
sys.path.insert(0, './')

import torch
import trimesh
import numpy as np
from libs.interaction import Interactions

device = "cuda:1"

obj = np.load("/data2/czk/Behave/behave-30fps-params-v1/Date01_Sub01_chairblack_sit/object_fit_all.npz")
smpls = np.load("/data2/czk/Behave/behave-30fps-params-v1/Date01_Sub01_chairblack_sit/smpl_fit_all.npz")

objects = trimesh.load('assets/chairblack/chairblack_center.obj', force='mesh')
center = np.mean(objects.vertices, 0)
print(center)
# objects.vertices -= center
# objects.export('assets/chairblack/chairblack_center.obj')

Sampler = Interactions(torch.zeros(1, 156), device=device)
Sampler.process_data(smpls, obj, objects, 'datasets/BEHAVE')