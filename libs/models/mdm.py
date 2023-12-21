import os
import torch
import inspect
import time
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from lightning.pytorch import LightningModule
from transformers import AutoModel, AutoTokenizer
from pytorch3d.transforms import rotation_6d_to_matrix
from libs.losses.geodesic_loss import geodesic_loss_R

from ..body_model import BodyModel
from ..get_model import get_model, get_dataset, instantiate_from_config
from .diffusion import create_diffusion
from .embedding import LabelEmbedder
# from .transform import recover_root_rot_pos, matrot2aa
# from .tgm_conversion import quaternion_to_angle_axis, rotation_6d_to_matrix
from .utils import copy2cpu


class MDM(LightningModule):
    """
    Latent Motion Diffusion
    """
    def __init__(self,
                 config,
                 datamodule) -> None:
        super().__init__()

        self.joints_num = config.MODEL.joints_num
        self.config = config

        self.denoiser = get_model(config.MODEL.DENOISER)

        # condition on action class
        # self.class_embedder = LabelEmbedder(config.MODEL.ACTION.num_classes, config.MODEL.DENOISER.args.hidden_size, config.MODEL.ACTION.class_dropout_prob)
        
        # condition on text

        if config.MODEL.DENOISER.args.hidden_size == config.MODEL.text_embed_dim:
            self.text_proj = None
        else:
            self.text_proj = torch.nn.Linear(config.MODEL.text_embed_dim, config.MODEL.DENOISER.args.hidden_size)

        if config.MODEL.DENOISER.args.hidden_size == self.text_embedder.text_embed_dim:
            self.text_proj = None
        else:
            self.text_proj = torch.nn.Linear(self.text_embedder.text_embed_dim, config.MODEL.DENOISER.args.hidden_size)

        self.num_inference_timesteps=config.MODEL.DENOISER.num_inference_timesteps
        
        if hasattr(config.MODEL, "VAE"):
            self.vae = get_model(
                config.MODEL.VAE
            )
            self.vae.load_state_dict(torch.load(config.MODEL.VAE.CHECKPOINT)['state_dict'])
            # freeze vae
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = None

        # self.diffusion = create_diffusion(
        #     timestep_respacing=config.MODEL.DENOISER.timestep_respacing, 
        #     predict_xstart=config.MODEL.DENOISER.predict_xstart,
        #     noise_schedule=config.MODEL.DENOISER.noise_schedule,
        #     learn_sigma=config.MODEL.DENOISER.args.learn_sigma,
        #     sigma_small=config.MODEL.DENOISER.sigma_small
        # )
            
        self.scheduler = instantiate_from_config(config.scheduler)
        self.noise_scheduler = instantiate_from_config(config.noise_scheduler)
        
        self.guidance_scale = 7.5
        self.guidance_uncodp = config.TRAIN.guidance_uncodp

        # build dataset
        self.dataset = datamodule

    def get_text_embedding(self, text_embeddings):
        """
        Generate Text Embedding with tokenizer and embedder
        """
        # text_inputs = self.tokenizer[0](
        #     text_embeddings,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.tokenizer[0].model_max_length,
        #     return_tensors="pt"
        # )
        # text_input_ids = text_inputs.input_ids
        # text_embeddings = self.text_embedder[0].get_text_features(
        #     text_input_ids.cpu()
        # )
        # (batch, text_dim) -> (batch, 1, text_dim)
        text_embeddings = torch.from_numpy(np.concatenate(text_embeddings)).float()
        if self.text_proj is not None:
            text_embeddings = self.text_proj(text_embeddings.to(self.text_proj.device))
        text_embeddings = text_embeddings.unsqueeze(1).to(self.device)
        return text_embeddings
    
    def generate_sequence_mask(self, length):
        max_length = max(length)
        mask = torch.arange(max_length).expand(len(length), max_length).to(length.device) < (length.unsqueeze(1))
        return mask
    
    @torch.no_grad()
    def sample(self, batch):
        # get input
        texts = batch["text"]
        lengths = torch.tensor(batch["length"], dtype=torch.int64)
        max_length = max(lengths)
        text_embedding = self.get_text_embedding(texts)
        attention_mask = self.generate_sequence_mask(lengths).to(text_embedding.device)
        
        # diffusion
        model_kwargs = dict(y=text_embedding, attention_mask=~attention_mask)
        z = self._diffusion_reverse(
            (len(batch["text"]), max_length, self.denoiser.output_size),
            model_kwargs
        )

        # decode
        if self.vae:
            z_body = z.view(-1, self.vae.latentD)
            pose_body_rec = self.vae.decode(z_body)['pose_body'].contiguous().view(len(batch["text"]), -1, (self.joints_num-1)*3)
        else:
            pose_body_rec = z

        return pose_body_rec
    
    def _compute_loss(self, mo_orig, mo_rec, attention_mask=None):
        """
        compute reconstruction loss with sample
        """
        g_loss = geodesic_loss_R()
        if attention_mask is not None:
            attention_mask = attention_mask.flatten()
        else:
            attention_mask = torch.ones(mo_orig.shape[0]*mo_orig.shape[1], dtype=torch.bool).to(mo_orig.device)
        l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
        # motion reconstraction

        rotation_loss = g_loss(
            rotation_6d_to_matrix(mo_orig.reshape(-1, self.joints_num, 6))[attention_mask, ...].view(-1, 3, 3),
            rotation_6d_to_matrix(mo_rec.reshape(-1, self.joints_num, 6))[attention_mask, ...].view(-1, 3, 3)
        )

        # # mask select
        # v2v = l1_loss(bm_rec.v, bm_orig.v)
        # j2j = l1_loss(bm_rec_local.Jtr, bm_orig_local.Jtr)
        # v2v_local = l1_loss(bm_rec_local.v, bm_orig_local.v)

        # mesh_rec = torch.sqrt(torch.pow(bm_rec.v-bm_orig.v, 2).sum(-1)).mean()
        # mesh_rec = torch.sqrt(torch.pow(bm_rec_local.v-bm_orig_local.v, 2).sum(-1)).mean()

        return {
            # "v2v":v2v, 
            # "j2j": j2j, 
            # "mesh_rec": mesh_rec, 
            # "v2v_local": v2v_local,
            "geodesic_loss": rotation_loss
        }


    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # get input
        add_index = [
            i for i in range(len(batch["text"])) if np.random.rand(1) < self.guidance_uncodp
        ]
        texts = batch["text"] + [self.dataset.text_zero_padding] * len(add_index)
        lengths =  torch.cat([batch["length"], batch["length"][add_index]])
        attention_mask = self.generate_sequence_mask(lengths)

        # full_pose = torch.cat([batch['pose_root'], batch['pose_body']], dim=-1)
        full_pose = batch['pose_body']

        # get latent
        with torch.no_grad():
            text_embedding = self.get_text_embedding(texts)
            if self.vae:
                joint_rot = batch['pose_body'].view(-1, (self.joints_num-1)*3)
                q_z_sample, dist = self.vae.encode(joint_rot)
                q_z_sample = q_z_sample.view(len(batch["text"]), -1, self.vae.latentD)
            else:
                q_z_sample = full_pose
        # latent_mask = attention_mask.unsqueeze(-1).repeat(1, 1, q_z_sample.shape[-1])
        # q_z_sample = q_z_sample * latent_mask
        
        # diffusion loss
        # x = torch.cat([root_pose, q_z_sample], dim=-1)
        x = torch.cat([q_z_sample, q_z_sample[add_index, ...]])
        
        model_kwargs = dict(y=text_embedding, attention_mask=~attention_mask)
        # loss_dict = self.diffusion.training_losses(self.denoiser, x, t, model_kwargs)

        n_set = self._diffusion_process(x, model_kwargs)
        # loss_mask = attention_mask.unsqueeze(-1).repeat(1, 1, n_set["pred"].shape[-1])
        diffusion_loss = nn.MSELoss(reduction='sum')(n_set["pred"], n_set["latent"]) / (x.shape[0] * x.shape[1])
        # diffusion_loss = torch.sum(diffusion_loss * loss_mask) / (torch.sum(lengths) * n_set["pred"].shape[-1])

        # reconstruction loss for condition outputs
        x_pred = n_set['pred']
        if self.vae:
            z_body = x_pred.view(-1, self.vae.latentD)
            pose_body_rec = self.vae.decode(z_body)['pose_body'].contiguous().view(len(texts), -1, (self.joints_num-1)*3)
        else:
            pose_body_rec = x_pred

        d_orig = torch.cat([full_pose, full_pose[add_index, ...]])
        rec_loss = self._compute_loss(d_orig, pose_body_rec, attention_mask)
        # rec_loss, _, _ = self._compute_loss(d_orig, pose_body_rec)
        
        loss = 0
        rec_loss['diff_loss'] = diffusion_loss
        for key, value in rec_loss.items():
            if hasattr(self.config.TRAIN.LOSS, key):
                loss = loss + getattr(self.config.TRAIN.LOSS, key) * value
        
        # log
        self.log_dict(rec_loss, prog_bar=True, logger=True, sync_dist=True)

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        texts = batch["text"] + [self.dataset.text_zero_padding] * len(batch["text"])
        lengths = torch.cat([batch["length"]] * 2)
        max_length = max(lengths)
        
        # full_pose = torch.cat([batch['pose_root'], batch['pose_body']], dim=-1)
        full_pose = batch['pose_body']
        
        with torch.no_grad():
            text_embedding = self.get_text_embedding(texts)
            attention_mask = self.generate_sequence_mask(lengths)
            
            # diffusion
            model_kwargs = dict(y=text_embedding, attention_mask=~attention_mask)
            # noise_mask = attention_mask.unsqueeze(-1).repeat(1, 1, self.denoiser.output_size)
            # noise = torch.randn(len(texts), self.denoiser.patch_size, self.denoiser.output_size).to(noise_mask.device)
            # noise = noise * noise_mask
            # z = self.diffusion.ddim_sample_loop(
            #     self.denoiser, 
            #     (len(texts), self.denoiser.patch_size, self.denoiser.output_size),
            #     model_kwargs=model_kwargs,
            #     clip_denoised=False,
            # )
            z = self._diffusion_reverse(
                (len(batch["text"]), max_length, self.denoiser.output_size),
                model_kwargs
            )

            # decode
            if self.vae:
                z_body = z.view(-1, self.vae.latentD)
                pose_body_rec = self.vae.decode(z_body)['pose_body'].contiguous().view(len(batch["text"]), -1, (self.joints_num-1)*3)
            else:
                pose_body_rec = z
            # z_rec_loss = l1_loss(d_rec, d_orig)
            rec_loss = self._compute_loss(full_pose, pose_body_rec, attention_mask[:len(batch["length"])])
            # rec_loss, body_rec, bm_orig = self._compute_loss(full_pose, pose_body_rec)
            # rec_loss.update({'angle_loss': z_rec_loss})
            val_loss = rec_loss['geodesic_loss']

            if batch_idx == 1 and (self.current_epoch+1) % 5 == 0:
                os.makedirs("./results-v1", exist_ok=True)
                body_rec = self.dataset.recover_motion(pose_body_rec[0, :batch["length"][0]].cpu())
                np.save('./results-v1/{}.npy'.format(self.current_epoch), body_rec.v.cpu().numpy())
                # np.save('./results/{}_origin.npy'.format(self.current_epoch), bm_orig.v[0:batch["length"][0], ..., [0, 2, 1]].cpu().numpy())
                # f = open('./results/{}.txt'.format(self.current_epoch), 'w')
                # f.write(batch["text"][0])
                # f.close()

        self.log_dict(rec_loss, prog_bar=True, logger=True, sync_dist=True)

        self.log("val_loss", val_loss, prog_bar=True, logger=True, sync_dist=True)
        return {'val_loss': copy2cpu(val_loss)}

    # def _get_data(self, split_name):

    #     dataset = get_dataset(self.config, split_name)

    #     return DataLoader(dataset,
    #                       batch_size=self.config.TRAIN.BATCH_SIZE,
    #                       shuffle=True if 'train' in split_name else False,
    #                       num_workers=self.config.TRAIN.NUM_WORKERS,
    #                       collate_fn=mld_collate,
    #                       pin_memory=True)

    # def train_dataloader(self):
    #     return self._get_data('train_val')
    
    # def val_dataloader(self):
    #     return self._get_data('val')

    def configure_optimizers(self):
        gen_params = [a[1] for a in self.denoiser.named_parameters() if a[1].requires_grad]
        gen_optimizer_class = getattr(optim_module, self.config.TRAIN.OPTIM.TYPE)
        gen_optimizer = gen_optimizer_class(gen_params, **self.config.TRAIN.OPTIM.ARGS)

        lr_sched_class = getattr(lr_sched_module, self.config.TRAIN.LR_SCHEDULER.TYPE)

        gen_lr_scheduler = lr_sched_class(gen_optimizer, **self.config.TRAIN.LR_SCHEDULER.ARGS)

        schedulers = [
            {
                'scheduler': gen_lr_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': self.config.TRAIN.val_frequency
            },
        ]
        return [gen_optimizer], schedulers
    
    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['state_dict'] = self.denoiser.state_dict()

    def _diffusion_process(self, latents, model_kwargs):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser(
            x=noisy_latents,
            t=timesteps,
            **model_kwargs
        )
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        n_set = {
            "noise": noise,
            "pred": noise_pred,
            "latent": latents
        }
        return n_set

    def _diffusion_reverse(self, shape, model_kwargs, noise=None, device=None):
        # init latents
        if device is None:
            device = next(self.denoiser.parameters()).device

        if noise:
            latents = noise.to(device)
        else:
            latents = torch.randn(
                shape,
                device=device,
                dtype=torch.float32,
            )
            # latents[model_kwargs['attention_mask'].unsqueeze(-1).repeat(1, 1, shape[-1])] = 0

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.config.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        # if "eta" in set(
        #         inspect.signature(self.noise_scheduler.step).parameters.keys()):
        #     extra_step_kwargs["eta"] = self.config.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) 
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            time = torch.ones((latent_model_input.shape[0],), device=t.device).long() * t
            noise_pred = self.denoiser(
                x=latent_model_input,
                t=time,
                **model_kwargs
            )
            # perform guidance
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample

        return latents

        