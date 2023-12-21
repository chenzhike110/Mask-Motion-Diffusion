import torch
import random
from os import path as osp

from torch.utils.data import DataLoader
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from lightning.pytorch import LightningModule

from .utils import make_deterministic, copy2cpu
from ..losses.geodesic_loss import geodesic_loss_R
from ..data.AMASS import VPoserDS
from ..body_model.body_model import BodyModel
from ..models.vposer import VPoser
from ..models.transform import aa2matrot

class VPoserTrainer(LightningModule):
    """
    Trainer wrapper for vposer and AMASS
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dataset_dir = self.config.DATASET.AMASS
        
        # set up model
        self.vposer = VPoser(
            config.MODEL.num_neurons,
            config.MODEL.latentD
        )

        # set up human smpl
        with torch.no_grad():
            self.bm_train = BodyModel(config.SMPL_PATH)

    def forward(self, joint6d):

        return self.vposer(joint6d)
    
    def _get_data(self, split_name):

        assert split_name in ('train', 'vald', 'test')

        dataset = VPoserDS(osp.join(self.dataset_dir, split_name), data_fields = ['pose_body'])

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
        gen_params = [a[1] for a in self.vposer.named_parameters() if a[1].requires_grad]
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
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.vposer.state_dict()