import clip
import torch
import wandb
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from pytorch_lightning import LightningModule
from pytorch3d.transforms import rotation_6d_to_matrix

from libs.smplx import SMPLHLayer
from libs.tools.parse import get_instance
from libs.losses.geodesic_loss import geodesic_loss_R

class MDM(LightningModule):
    def __init__(
        self, 
        cfg,
        denoiser,
        scheduler,
        text_encoder
    ):
        super().__init__()

        self.cfg = cfg
        
        self.denoiser = denoiser
        self.scheduler = scheduler
        
        self.cond_drop_prob = cfg.train.cond_drop_prob
        self.guidance_scale = cfg.scheduler.guidance_scale
        self.uncondition_embedding = nn.parameter.Parameter(torch.zeros(1, 1, 512, dtype=torch.float32))
        self.padding_token = nn.parameter.Parameter(torch.zeros(1, 22*9, dtype=torch.float32))

        self.text_encoder = text_encoder
        self.init_text_encoder(text_encoder)

        self.body_model = SMPLHLayer(model_path=cfg.smpl_path)
        self.pose_index = 3

    def padding_input(self, x, length):
        padding = self.padding_token.repeat(x.shape[0], x.shape[1], 1)
        for index, data in enumerate(x):
            padding[index, length[index]:, :] = padding
        return padding

    def init_text_encoder(self, name):
        """
        Initialize text encoder from name.
        """
        if 'clip' in name:
            clip_model, clip_preprocess = clip.load(name.strip('clip:'), device='cpu',
                                                    jit=False)  # Must set jit=False for training
            # Cannot run on cpu
            clip.model.convert_weights(clip_model) 

            # Freeze CLIP weights
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad = False

            def encode_text(raw_text):
                device = next(self.parameters()).device
                text = clip.tokenize(raw_text, truncate=True).to(device)
                feat_clip_text = self.text_encoder.encode_text(text).float()
                return feat_clip_text

            self.encode_text = encode_text

        elif 'bert' in name:
            assert NotImplementedError
        
        else:
            def encode_text(text):
                device = next(self.parameters()).device
                return torch.cat(text).to(device).unsqueeze(1)
            
            self.encode_text = encode_text

    def generate_attention_mask(self, length, trunck_size=0):
        max_length = max(max(length), trunck_size)
        mask = torch.arange(max_length).expand(len(length), max_length).to(length.device) < (length.unsqueeze(1))
        return mask
    
    def recover_motion(self, data):
        # 22 * 9
        if data.shape[-1] == 198:
            transl = data[..., :3].reshape(-1, 3)
            # transl[..., 0] *= -1
            body_pose = rotation_6d_to_matrix(data[...,22*3:].reshape(-1, 22, 6))
        elif data.shape[-1] == 201:
            transl = data[..., 22*3:23*3].reshape(-1, 3)
            # transl[..., 0] *= -1
            body_pose = rotation_6d_to_matrix(data[..., 23*3:].reshape(-1, 22, 6))
        elif data.shape[-1] == 135: # 22*6+3
            transl = data[..., :3].reshape(-1, 3)
            # transl[..., 0] *= -1
            body_pose = rotation_6d_to_matrix(data[..., 3:].reshape(-1, 22, 6))
        else:
            assert NotImplementedError

        body = self.body_model(body_pose=body_pose[:, 1:], global_orient=body_pose[:, 0], transl=transl)
        return body
    
    def _compute_loss(self, n_set, attention_mask):
        g_loss = geodesic_loss_R(reduction='sum')
        d_loss = nn.L1Loss(reduction='none')
        j_loss = nn.L1Loss(reduction='none')

        bs, n_token, latent_dim = n_set["latent"].shape

        body_pose_orig = rotation_6d_to_matrix(n_set["latent"][..., self.pose_index:].reshape(-1, 22, 6)[attention_mask.flatten()]).view(-1, 3, 3)
        body_pose_rec = rotation_6d_to_matrix(n_set["pred"][..., self.pose_index:].reshape(-1, 22, 6)[attention_mask.flatten()]).view(-1, 3, 3)
        rotation_loss = g_loss(body_pose_rec, body_pose_orig) / (torch.sum(attention_mask) * 22)

        transl_orig = n_set["latent"][..., :3].reshape(-1, 3)[attention_mask.flatten()]
        transl_rec = n_set["pred"][..., :3].reshape(-1, 3)[attention_mask.flatten()]

        body_pose_orig = body_pose_orig.reshape(-1, 22, 3, 3)
        body_pose_rec = body_pose_rec.reshape(-1, 22, 3, 3)
        
        body_orig = self.body_model(body_pose=body_pose_orig[:, 1:], global_orient=body_pose_orig[:, 0], transl=transl_orig, return_verts=False)
        body_rec = self.body_model(body_pose=body_pose_rec[:, 1:], global_orient=body_pose_rec[:, 0], transl=transl_rec, return_verts=False)
        joint_loss = j_loss(body_orig.joints, body_rec.joints)[:, 1:22, ...].sum(dim=(1, 2)).mean() / 21.0
        # joint_loss = 0.0

        diffusion_loss = d_loss(n_set["pred"][..., :3], n_set["latent"][..., :3])
        diffusion_loss = diffusion_loss.sum(dim=-1)
        diffusion_loss = torch.sum(diffusion_loss * attention_mask) / (torch.sum(attention_mask))

        return {
            "diff_loss": diffusion_loss,
            "rot_loss": rotation_loss,
            "jp_loss": joint_loss,
            "loss": self.cfg.train.loss.diff_loss * diffusion_loss + \
                  self.cfg.train.loss.geodesic_loss * rotation_loss + \
                  self.cfg.train.loss.pos_loss * joint_loss ,
        }

    @torch.no_grad()
    def sample(self, batch):
        """
        sample from the model
        """
        text_embedding = self.encode_text(batch["text"])
        lengths = torch.tensor(batch["length"] * 2, dtype=torch.int64).to(self.device)
        attention_mask = self.generate_attention_mask(lengths)
        text_embedding = torch.cat([text_embedding, self.uncondition_embedding.repeat(len(batch["text"]), 1, 1)])
        max_length = max(lengths)

        model_kwargs = dict(cond=text_embedding, mask=~attention_mask)
        z = self._diffusion_reverse(
            (len(batch["text"]), max_length, self.denoiser.output_size),
            model_kwargs,
            visual=True
        )

        return z
    
    @torch.no_grad()
    def inpainting(self, batch):
        """
        inpainting motion with part trajectory
        """
        # text_embedding = self.encode_text(batch["text"])
        lengths = torch.tensor(batch["length"] * 2, dtype=torch.int64).to(self.device)
        attention_mask = self.generate_attention_mask(lengths)

        if 'text' in batch.keys():
            text_embedding = self.encode_text(batch["text"])
            text_embedding = torch.cat([text_embedding, self.uncondition_embedding.repeat(len(batch["text"]), 1, 1)])
        else:
            text_embedding = self.uncondition_embedding.repeat(batch['motion'].shape[0] * 2, 1, 1)

        model_kwargs = dict(cond=text_embedding, mask=~attention_mask)

        motion = batch['motion'].repeat(2, 1, 1).to(self.device)
        inpainting_mask = batch['inpainting_mask'].repeat(2, 1, 1).to(self.device)
        z = self._diffusion_reverse(
            motion.shape,
            model_kwargs,
            noise=motion, 
            inpainting_mask=inpainting_mask,
            visual=True
        )

        return z

    def training_step(self, batch, batch_idx):
        """
        training step for lightning
        """
        add_index = random.sample(range(len(batch["text"])), int(self.cond_drop_prob * len(batch["text"])))
        texts = batch["text"]
        text_embedding = self.encode_text(texts)

        if type(batch['motion']) == list:
            batch['motion'] = self.padding_input(batch['motion'], batch["length"])
        
        text_embedding[add_index, ...] = self.uncondition_embedding
        lengths = batch["length"]
        x = batch['motion']

        attention_mask = self.generate_attention_mask(lengths, x.shape[1])
        # model_kwargs = dict(cond=text_embedding, mask=~attention_mask)
        model_kwargs = dict(cond=text_embedding)

        n_set = self._diffusion_process(x, model_kwargs)

        rec_loss = self._compute_loss(n_set, attention_mask)
        rec_loss = {
            "train/"+k: v for k, v in rec_loss.items()
        }

        self.log_dict(rec_loss, logger=True)
        return {'loss': rec_loss['train/loss']}
    
    def validation_step(self, batch, batch_idx):
        """
        training step for lightning
        """
        texts = batch["text"]
        lengths = batch["length"]
        text_embedding = self.encode_text(texts)

        if type(batch['motion']) == list:
            batch['motion'] = self.padding_input(batch['motion'], batch["length"])

        x = batch['motion']
        attention_mask = self.generate_attention_mask(lengths, x.shape[1])
        # model_kwargs = dict(cond=text_embedding, mask=~attention_mask)
        model_kwargs = dict(cond=text_embedding)

        with torch.no_grad():
            z = self._diffusion_reverse(
                (len(batch["text"]), batch['motion'].shape[1], self.denoiser.output_size),
                model_kwargs,
                guidance=False
            )

            if batch_idx == 0:
                downsample = 10
                smpl_output = self.recover_motion(z[0, :min(lengths[0], 140)][::downsample])
                color = torch.arange(0, smpl_output.vertices.shape[0])
                color = color.reshape(-1, 1, 1).repeat(1, smpl_output.vertices.shape[1], 1)
                _3dPoints = torch.cat([smpl_output.vertices.cpu(), color], dim=-1).view(-1, 4)
                wandb.log({"Generated Motion": [
                    wandb.Object3D(_3dPoints.numpy())
                ]})

            n_set = {
                "pred": z,
                "latent": x
            }
            rec_loss = self._compute_loss(n_set, attention_mask)
            rec_loss = {
                "val/"+k: v.cpu() for k, v in rec_loss.items()
            }

            self.log("val_loss", rec_loss['val/loss'])

        self.log_dict(rec_loss, logger=True)
        
        return {'loss': rec_loss['val/loss']}
    
    def on_validation_epoch_end(self) -> None:
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        from torch import optim as optim_module
        from torch.optim import lr_scheduler as lr_sched_module

        gen_params = [a[1] for a in self.denoiser.named_parameters() if a[1].requires_grad]
        gen_optimizer_class = getattr(optim_module, self.cfg.train.optimizer.target)
        gen_optimizer = gen_optimizer_class(gen_params, **self.cfg.train.optimizer.params)

        lr_sched_class = getattr(lr_sched_module, self.cfg.train.lr_scheduler.target)

        gen_lr_scheduler = lr_sched_class(gen_optimizer, **self.cfg.train.lr_scheduler.params)

        schedulers = [
            {
                'scheduler': gen_lr_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': self.cfg.train.val_frequency
            },
        ]
        return [gen_optimizer], schedulers
    
    def optimizer_step(self, epoch_nb, batch_nb, optimizer, opt_closure):
        if self.trainer.global_step < self.cfg.train.warm_up_iter:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.cfg.train.warm_up_iter))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.cfg.train.optimizer.params.lr

        optimizer.step(closure=opt_closure)

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
            self.scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Remove noise on the trajectory
        # noisy_latents[:, :, 3:22*3] = latents[:, :, 3:22*3]
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

    def _diffusion_reverse(self, shape, model_kwargs, noise=None, inpainting_mask=None, visual=False, guidance=True):
        # init latents
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
        
        if not inpainting_mask:
            inpainting_mask = torch.zeros_like(latents).bool()

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        # if "eta" in set(
        #         inspect.signature(self.noise_scheduler.step).parameters.keys()):
        #     extra_step_kwargs["eta"] = self.config.scheduler.eta

        if visual:
            bar = tqdm(timesteps)
        else:
            bar = timesteps
        # reverse
        for t in bar:
            # expand the latents if we are doing classifier free guidance
            if guidance:
                latent_model_input = torch.cat([latents] * 2) 
            else:
                latent_model_input = latents.clone()
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            time = torch.ones((latent_model_input.shape[0],), device=t.device).long() * t
            noise_pred = self.denoiser(
                x=latent_model_input,
                t=time,
                **model_kwargs
            )
            if guidance:
            # perform guidance
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            latents_pre = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            latents[~inpainting_mask] = latents_pre[~inpainting_mask]

        return latents

    @classmethod
    def from_config(cls, cfg):
        """
        Initialize a model from a config.
        """
        
        text_encoder = cfg.get('text_encoder', 'ViT-B/32')
        denoiser = get_instance(cfg.get('denoiser'))
        scheduler = get_instance(cfg.get('scheduler'))

        model = cls(
            cfg=cfg,
            denoiser=denoiser,
            scheduler=scheduler,
            text_encoder=text_encoder,
        )

        return model


class Trajectory_MDM(MDM):
    """
    Trajectory Following MDM
    """
    def __init__(self, cfg, denoiser, scheduler, text_encoder):
        super().__init__(cfg, denoiser, scheduler, text_encoder)

        self.joint_num = 22
        self.cond_trajectory_prob = cfg.train.get('cond_trajectory_prob', 0.5)
        # self.uncondition_trajectory_embedding = nn.parameter.Parameter(torch.zeros(1, 1, (self.joint_num - 1)*3))

    def mask_ref_trajectory(self, motion):
        trajectory_mask = torch.rand(motion.shape[0], motion.shape[1], (self.joint_num - 1)*3) < self.cond_trajectory_prob
        motion[..., 3:self.joint_num*3][~trajectory_mask] = 0.
        return trajectory_mask

    def training_step(self, batch, batch_idx):
        """
        training step for lightning
        """
        # texts = batch["text"]
        lengths = batch["length"]

        # text_embedding = self.encode_text(texts)
        # add_index = random.sample(range(len(batch["text"])), int(self.cond_drop_prob * len(batch["text"])))        

        if type(batch['motion']) == list:
            batch['motion'] = self.padding_input(batch['motion'], batch["length"])

        text_embedding = self.uncondition_embedding.repeat(batch["length"].shape[0], 1, 1)
        x = batch['motion']

        attention_mask = self.generate_attention_mask(lengths, x.shape[1])
        model_kwargs = dict(cond=text_embedding)
        n_set = self._diffusion_process(x, model_kwargs)

        rec_loss = self._compute_loss(n_set, attention_mask)
        rec_loss = {
            "train/"+k: v for k, v in rec_loss.items()
        }

        self.log_dict(rec_loss, prog_bar=True, logger=True)
        return {'loss': rec_loss['loss']}

    def _compute_loss(self, n_set, attention_mask, trajectory_mask):
        g_loss = geodesic_loss_R(reduction='sum')
        d_loss = nn.L1Loss(reduction='none')
        j_loss = nn.L1Loss(reduction='none')

        bs, n_token, latent_dim = n_set["latent"].shape

        body_pose_orig = rotation_6d_to_matrix(n_set["latent"][..., self.pose_index:].reshape(-1, 22, 6)[attention_mask.flatten()]).view(-1, 3, 3)
        body_pose_rec = rotation_6d_to_matrix(n_set["pred"][..., self.pose_index:].reshape(-1, 22, 6)[attention_mask.flatten()]).view(-1, 3, 3)
        rotation_loss = g_loss(body_pose_rec, body_pose_orig) / (torch.sum(attention_mask) * 22)

        transl_orig = n_set["latent"][..., :3].reshape(-1, 3)[attention_mask.flatten()]
        transl_rec = n_set["pred"][..., :3].reshape(-1, 3)[attention_mask.flatten()]

        body_pose_orig = body_pose_orig.reshape(-1, 22, 3, 3)
        body_pose_rec = body_pose_rec.reshape(-1, 22, 3, 3)
        
        body_orig = self.body_model(body_pose=body_pose_orig[:, 1:], global_orient=body_pose_orig[:, 0], transl=transl_orig, return_verts=False)
        body_rec = self.body_model(body_pose=body_pose_rec[:, 1:], global_orient=body_pose_rec[:, 0], transl=transl_rec, return_verts=False)
        joint_loss = j_loss(body_orig.joints, body_rec.joints)[:, 1:22, ...].sum(dim=(1, 2)).mean() / 21.0
        # joint_loss = 0.0

        diffusion_loss = d_loss(n_set["pred"][..., :3], n_set["latent"][..., :3])
        diffusion_loss = diffusion_loss.sum(dim=-1)
        diffusion_loss = torch.sum(diffusion_loss * attention_mask) / (torch.sum(attention_mask))

        return {
            "diff_loss": diffusion_loss,
            "rot_loss": rotation_loss,
            "jp_loss": joint_loss,
            "loss": self.cfg.train.loss.diff_loss * diffusion_loss + \
                  self.cfg.train.loss.geodesic_loss * rotation_loss + \
                  self.cfg.train.loss.pos_loss * joint_loss,
        }