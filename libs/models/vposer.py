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
import os
import numpy as np
import torch
from .transform import matrot2aa, aa2matrot
from .utils import copy2cpu
from .tgm_conversion import angle_axis_to_rotation_matrix, rotation_matrix_to_rotation_6d
from ..get_model import get_dataset
from ..losses.geodesic_loss import geodesic_loss_R
from ..body_model.body_model import BodyModel
from torch import nn
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule


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
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, Xout):
        mu = self.mu(Xout)
        var = torch.exp(0.5 * self.logvar(Xout))
        eps = self.N.sample(var.shape).to(var.device)
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

class VPoser(LightningModule):
    """
    Change from euler to rot6d
    """
    def __init__(self,
                 config=None, 
                 num_neurons : int = 512, 
                 latentD : int = 32):
        super(VPoser, self).__init__()

        self.config = config
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
        if self.config:
            with torch.no_grad():
                self.bm_train = BodyModel(config.SMPL_PATH)
    
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
    
    def _get_data(self, split_name):

        assert split_name in ('train', 'vald', 'test')
        dataset = get_dataset(self.config, split_name)

        assert len(dataset) != 0, ValueError('Dataset has nothing in it!')
        
        return DataLoader(dataset,
                          batch_size=self.config.TRAIN.BATCH_SIZE,
                          shuffle=True if split_name == 'train' else False,
                          num_workers=self.config.TRAIN.NUM_WORKERS,
                          pin_memory=True)
    
    def train_dataloader(self):
        return self._get_data('train')
    
    def val_dataloader(self):
        return self._get_data('vald')
    
    def configure_optimizers(self):
        gen_params = [a[1] for a in self.named_parameters() if a[1].requires_grad]
        gen_optimizer_class = getattr(optim_module, self.config.TRAIN.OPTIM.TYPE)
        gen_optimizer = gen_optimizer_class(gen_params, **self.config.TRAIN.OPTIM.ARGS)

        lr_sched_class = getattr(lr_sched_module, self.config.TRAIN.LR_SCHEDULER.TYPE)

        gen_lr_scheduler = lr_sched_class(gen_optimizer, **self.config.TRAIN.LR_SCHEDULER.ARGS)

        schedulers = [
            {
                'scheduler': gen_lr_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            },
        ]
        return [gen_optimizer], schedulers
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):

        drec = self(batch['pose_body'].view(-1, 63))

        loss = self._compute_loss(batch, drec)

        train_loss = loss['weighted_loss']['loss_total']

        for k, v in loss['weighted_loss'].items():
            self.log(k, v, prog_bar=True, logger=True)

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):

        drec = self(batch['pose_body'].view(-1, 63))

        loss = self._compute_loss(batch, drec)
        val_loss = loss['unweighted_loss']['loss_total']

        self.log("val_loss", val_loss)
        return {'val_loss': copy2cpu(val_loss)}

    def _compute_loss(self, dorig, drec):
        l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
        geodesic_loss = geodesic_loss_R(reduction='mean')

        bs, latentD = drec['poZ_body_mean'].shape
        device = drec['poZ_body_mean'].device

        loss_kl_wt = self.config.LOSS.loss_kl_wt
        loss_rec_wt = self.config.LOSS.loss_rec_wt
        loss_matrot_wt = self.config.LOSS.loss_matrot_wt
        loss_jtr_wt = self.config.LOSS.loss_jtr_wt

        # q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])
        q_z = drec['q_z']
        # dorig['fullpose'] = torch.cat([dorig['root_orient'], dorig['pose_body']], dim=-1)

        # Reconstruction loss - L1 on the output mesh
        with torch.no_grad():
            bm_orig = self.bm_train(pose_body=dorig['pose_body'])

        bm_rec = self.bm_train(pose_body=drec['pose_body'].contiguous().view(bs, -1))

        v2v = l1_loss(bm_rec.v, bm_orig.v)

        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((bs, latentD), device=device, requires_grad=False),
            scale=torch.ones((bs, latentD), device=device, requires_grad=False))
        weighted_loss_dict = {
            'loss_kl':loss_kl_wt * torch.distributions.kl.kl_divergence(q_z, p_z).mean(),
            'loss_mesh_rec': loss_rec_wt * v2v
        }

        weighted_loss_dict['matrot'] = loss_matrot_wt * geodesic_loss(drec['pose_body_matrot'].view(-1,3,3), aa2matrot(dorig['pose_body'].view(-1, 3)))
        weighted_loss_dict['jtr'] = loss_jtr_wt * l1_loss(bm_rec.Jtr, bm_orig.Jtr)

        weighted_loss_dict['loss_total'] = torch.stack(list(weighted_loss_dict.values())).sum()

        # if (self.current_epoch < self.config.TRAIN.keep_extra_loss_terms_until_epoch):
            # breakpoint()

        with torch.no_grad():
            unweighted_loss_dict = {'v2v': torch.sqrt(torch.pow(bm_rec.v-bm_orig.v, 2).sum(-1)).mean()}
            unweighted_loss_dict['loss_total'] = torch.cat(
                list({k: v.view(-1) for k, v in unweighted_loss_dict.items()}.values()), dim=-1).sum().view(1)

        return {'weighted_loss': weighted_loss_dict, 'unweighted_loss': unweighted_loss_dict}
