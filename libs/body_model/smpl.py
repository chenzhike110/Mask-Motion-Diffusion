# This code is based on https://github.com/Mathux/ACTOR.git
import numpy as np
import torch

import contextlib

from smplx import SMPLLayer as _SMPLLayer
from smplx.lbs import vertices2joints


# adapted from VIBE/SPIN to output smpl_joints, vibe joints and action2motion joints
class SMPL(_SMPLLayer):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, model_path, **kwargs):
        kwargs["model_path"] = model_path

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(**kwargs)
            
        
    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        all_joints = torch.cat([smpl_output.joints, extra_joints], dim=1)

        output = {"vertices": smpl_output.vertices}

        for joinstype, indexes in self.maps.items():
            output[joinstype] = all_joints[:, indexes]
            
        return output