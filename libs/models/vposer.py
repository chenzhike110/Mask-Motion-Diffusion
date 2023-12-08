# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12

import numpy as np
import torch
from .transform import matrot2aa
from .tgm_conversion import angle_axis_to_rotation_matrix, rotation_matrix_to_rotation_6d
from torch import nn
from torch.nn import functional as F


class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(x.shape[0], -1)

class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        mu = self.mu(Xout)
        var = torch.exp(0.5 * self.logvar(Xout))
        eps = torch.randn_like(var)
        return eps * var + mu, torch.distributions.normal.Normal(mu, var)

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)

class VPoser(nn.Module):
    """
    Change from euler to rot6d
    """
    def __init__(self, 
                 num_neurons : int = 512, 
                 latentD : int = 32):
        super(VPoser, self).__init__()

        self.latentD = latentD

        self.num_joints = 21
        n_features = self.num_joints * 6

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.Linear(n_features, num_neurons),
            nn.BatchNorm1d(num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.BatchNorm1d(num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.BatchNorm1d(num_neurons),
            nn.LeakyReLU(),
            NormalDistDecoder(num_neurons, self.latentD)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.BatchNorm1d(num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.BatchNorm1d(num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder()
        )

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if isinstance(p, nn.Linear):
                torch.nn.init.kaiming_uniform_(p)

    def encode(self, pose_body):
        '''
        :param Pin: Nx(numjoints*6)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        # euler to rotation6d
        if pose_body.shape[-1] == self.num_joints * 3:
            batch, _ = pose_body.shape
            pose_body = pose_body.reshape(-1, 3)
            pose_body = rotation_matrix_to_rotation_6d(angle_axis_to_rotation_matrix(pose_body)[:, :3, :3]).reshape(batch, -1)
        return self.encoder_net(pose_body)

    def decode(self, Zin):
        bs = Zin.shape[0]

        prec = self.decoder_net(Zin)

        return {
            'pose_body': matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }


    def forward(self, pose_body):
        '''
        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''
        assert torch.isnan(pose_body).any().sum() == 0
        q_z_sample, q_z = self.encode(pose_body)
        # q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)
        decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z})
        return decode_results

    def sample_poses(self, num_poses, seed=None):
        np.random.seed(seed)

        some_weight = [a for a in self.parameters()][0]
        dtype = some_weight.dtype
        device = some_weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype, device=device)

        return self.decode(Zgen)
