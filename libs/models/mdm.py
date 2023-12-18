import torch
import random
from torch.utils.data import DataLoader
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from lightning.pytorch import LightningModule
from transformers import AutoModel, AutoTokenizer

from ..body_model import BodyModel
from ..get_model import get_model, get_dataset
from .diffusion import create_diffusion
from .embedding import LabelEmbedder
# from .transform import recover_root_rot_pos, matrot2aa
# from .tgm_conversion import quaternion_to_angle_axis, rotation_6d_to_matrix
from .utils import make_deterministic, copy2cpu


class MDM(LightningModule):
    """
    Latent Motion Diffusion
    """
    def __init__(self,
                 config) -> None:
        super().__init__()

        self.joints_num = config.MODEL.joints_num
        self.config = config

        self.denoiser = get_model(config.MODEL.DENOISER)

        # condition on action class
        self.class_embedder = LabelEmbedder(config.MODEL.ACTION.num_classes, config.MODEL.DENOISER.args.hidden_size, config.MODEL.ACTION.class_dropout_prob)
        
        # condition on text
        self.text_embedder = AutoModel.from_pretrained(config.MODEL.TEXT.ENCODER)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT.TOKENIZER)

        if config.MODEL.DENOISER.args.hidden_size == self.text_embedder.text_embed_dim:
            self.text_proj = None
        else:
            self.text_proj = torch.nn.Linear(self.text_embedder.text_embed_dim, config.MODEL.DENOISER.args.hidden_size)

        self.num_inference_timesteps=config.MODEL.DENOISER.num_inference_timesteps
        
        self.vae = get_model(
            config.MODEL.VAE
        )
        self.vae.load_state_dict(torch.load(config.MODEL.VAE.CHECKPOINT)['state_dict'])
        
        # freeze vae
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # freeze text model
        for param in self.text_embedder.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            self.bm = BodyModel(config.SMPL_PATH)

        self.diffusion = create_diffusion(
            timestep_respacing=config.MODEL.DENOISER.timestep_respacing, 
            predict_xstart=config.MODEL.DENOISER.predict_xstart,
            noise_schedule=config.MODEL.DENOISER.noise_schedule,
            learn_sigma=config.MODEL.DENOISER.args.learn_sigma,
            sigma_small=config.MODEL.DENOISER.sigma_small
        )

        # build dataset
        dataset = get_dataset(self.config, [])
        self.register_buffer('Mean', torch.from_numpy(dataset.root_mean))
        self.register_buffer('Std', torch.from_numpy(dataset.root_std))

    def get_text_embedding(self, texts):
        """
        Generate Text Embedding with tokenizer and embedder
        """
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = self.text_embedder.get_text_features(
            text_input_ids.to(self.text_embedder.device)
        )
        # (batch, text_dim) -> (batch, 1, text_dim)
        if self.text_proj is not None:
            text_embeddings = self.text_proj(text_embeddings)
        text_embeddings = text_embeddings.unsqueeze(1)
        return text_embeddings
    
    def generate_sequence_mask(self, length):
        mask = torch.arange(self.denoiser.patch_size).expand(len(length), self.denoiser.patch_size).to(length.device) < (length.unsqueeze(1))
        return mask
    
    @torch.no_grad()
    def sample(self, batch):
        # get input
        texts = batch["text"]
        lengths = torch.tensor(batch["length"], dtype=torch.int64)
        text_embedding = self.get_text_embedding(texts)
        attention_mask = self.generate_sequence_mask(lengths).to(text_embedding.device)
        
        # diffusion
        model_kwargs = dict(y=text_embedding, attention_mask=attention_mask)
        z = self.diffusion.ddim_sample_loop(
            self.denoiser, 
            (len(texts), self.denoiser.patch_size, self.denoiser.output_size),
            model_kwargs=model_kwargs,
            clip_denoised=False,
        )

        # decode
        z_body = z[:, :, 6:].view(-1, self.vae.latentD)
        pose_body_rec = self.vae.decode(z_body)['pose_body'].contiguous().view(len(texts), -1, (self.joints_num-1)*3)
        pose_root_rec = z[:, :, :6] * self.Std + self.Mean
        d_rec = torch.cat([pose_root_rec, pose_body_rec], dim=-1)
        mo_rec = self._reconstruc_motion(d_rec)

        attention_mask = attention_mask[:, 1:].flatten()
        mo_rec = mo_rec.view(-1, mo_rec.shape[-1])[attention_mask, :]
        bm_rec = self.bm(pose_body=mo_rec[:, 6:], root_orient=mo_rec[:, 3:6], trans=mo_rec[:, :3])

        return bm_rec
    
    def _reconstruc_motion(self, data):
        root_pos = torch.cumsum(data[..., :2], dim=-2)
        # root_pos = torch.cat((root_pos, data[..., 2:3]), dim=-1)
        return torch.cat([root_pos, data[..., 2:]], dim=-1)
    
    def _compute_loss(self, dorig, drec, attention_mask):
        """
        compute all loss with sample
        origin: root 4 + body 6*joint_num
        result: root 4 + body 3*joint_num
        """
        attention_mask = attention_mask.flatten()
        l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
        # motion reconstraction
        with torch.no_grad():
            mo_orig = self._reconstruc_motion(dorig)
            mo_orig = mo_orig.view(-1, mo_orig.shape[-1])[attention_mask, :]
            bm_orig = self.bm(pose_body=mo_orig[:, 6:], root_orient=mo_orig[:, 3:6], trans=mo_orig[:, :3])
            bm_orig_local = self.bm(pose_body=mo_orig[:, 6:])

        mo_rec = self._reconstruc_motion(drec)
        mo_rec = mo_rec.view(-1, mo_rec.shape[-1])[attention_mask, :]
        bm_rec = self.bm(pose_body=mo_rec[:, 6:], root_orient=mo_rec[:, 3:6], trans=mo_rec[:, :3])
        bm_rec_local = self.bm(pose_body=mo_rec[:, 6:])

        # mask select
        v2v = l1_loss(bm_rec.v, bm_orig.v)
        v2v_local = l1_loss(bm_rec_local.v, bm_orig_local.v)
        j2j = l1_loss(bm_rec.Jtr, bm_orig.Jtr)

        mesh_rec = torch.sqrt(torch.pow(bm_rec.v-bm_orig.v, 2).sum(-1)).mean()

        return {"v2v": v2v, "j2j": j2j, "v2v_local": v2v_local, "mesh_rec": mesh_rec}


    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # get input
        texts = batch["text"]
        # uncondition
        texts = [
            "" if random.random() < self.config.TRAIN.guidance_uncodp else i
            for i in texts[:-1]
        ]
        texts.append(batch["text"][-1])

        lengths = batch["length"]
        joint_rot = batch['pose_body'].view(-1, (self.joints_num-1)*3)
        root_pose = batch['pose_root']

        text_embedding = self.get_text_embedding(texts)
        attention_mask = self.generate_sequence_mask(lengths)

        recontruction_loss_mask = [len(text)==0 for text in texts]
        recontruction_loss_mask = torch.tensor(recontruction_loss_mask).nonzero()

        # get latent
        q_z_sample, dist = self.vae.encode(joint_rot)
        q_z_sample = q_z_sample.view(len(texts), -1, self.vae.latentD)
        # latent_mask = attention_mask.unsqueeze(-1).repeat(1, 1, q_z_sample.shape[-1])
        # q_z_sample = q_z_sample * latent_mask
        
        # diffusion loss
        # x = torch.cat([root_pose, q_z_sample], dim=-1)
        x = q_z_sample

        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=joint_rot.device)
        model_kwargs = dict(y=text_embedding, attention_mask=attention_mask)
        loss_dict = self.diffusion.training_losses(self.denoiser, x, t, model_kwargs)
        # loss_mask = attention_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        # diffusion_loss = (loss_dict["target"] - loss_dict["model_output"]) ** 2
        # diffusion_loss = torch.sum(diffusion_loss * loss_mask) / torch.sum(loss_mask)
        diffusion_loss = loss_dict["mse"].mean()

        # reconstruction loss for condition outputs
        x_pred = loss_dict['model_output']
        z_body = x_pred.view(-1, self.vae.latentD)
        # z_body = x_pred[:, :, 6:].view(-1, self.vae.latentD)

        pose_body_rec = self.vae.decode(z_body)['pose_body'].contiguous().view(len(texts), -1, (self.joints_num-1)*3)
        
        # pose_root_rec = x_pred[:, :, :6] * self.Std + self.Mean
        # pose_root_ori = batch['pose_root'] * self.Std + self.Mean
        pose_root_rec = torch.zeros_like(batch['pose_root']) 
        pose_root_ori = torch.zeros_like(batch['pose_root'])

        d_orig = torch.cat([pose_root_ori, batch["pose_body"]], dim=-1)
        d_rec = torch.cat([pose_root_rec, pose_body_rec], dim=-1)

        attention_mask[recontruction_loss_mask, ...] = False
        rec_loss = self._compute_loss(d_orig, d_rec, attention_mask)
        
        loss = 0
        rec_loss['diff_loss'] = diffusion_loss
        for key, value in rec_loss.items():
            if hasattr(self.config.TRAIN.LOSS, key):
                loss = loss + getattr(self.config.TRAIN.LOSS, key) * value
        
        # log
        self.log_dict(rec_loss, prog_bar=True, logger=True, sync_dist=True)

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        l1_loss = torch.nn.SmoothL1Loss()
        texts = batch["text"]
        lengths = batch["length"]
        # pose_root_ori = batch['pose_root'] * self.Std + self.Mean
        pose_root_ori = torch.zeros_like(batch['pose_root'])
        d_orig = torch.cat([pose_root_ori, batch["pose_body"]], dim=-1)
        
        with torch.no_grad():
            text_embedding = self.get_text_embedding(texts)
            attention_mask = self.generate_sequence_mask(lengths)
            
            # diffusion
            model_kwargs = dict(y=text_embedding, attention_mask=attention_mask)
            # noise_mask = attention_mask.unsqueeze(-1).repeat(1, 1, self.denoiser.output_size)
            # noise = torch.randn(len(texts), self.denoiser.patch_size, self.denoiser.output_size).to(noise_mask.device)
            # noise = noise * noise_mask
            z = self.diffusion.ddim_sample_loop(
                self.denoiser, 
                (len(texts), self.denoiser.patch_size, self.denoiser.output_size),
                model_kwargs=model_kwargs,
                clip_denoised=False,
            )

            # decode
            z_body = z.view(-1, self.vae.latentD)
            pose_body_rec = self.vae.decode(z_body)['pose_body'].contiguous().view(len(texts), -1, (self.joints_num-1)*3)
            
            # pose_root_rec = z[:, :, :6] * self.Std + self.
            pose_root_rec = torch.zeros_like(batch['pose_root']) 
            d_rec = torch.cat([pose_root_rec, pose_body_rec], dim=-1)

            # z_rec_loss = l1_loss(d_rec, d_orig)

            rec_loss = self._compute_loss(d_orig, d_rec, attention_mask)
            # rec_loss.update({'angle_loss': z_rec_loss})
            val_loss = rec_loss['mesh_rec']

        self.log_dict(rec_loss, prog_bar=True, logger=True, sync_dist=True)

        self.log("val_loss", val_loss, prog_bar=True, logger=True, sync_dist=True)
        return {'val_loss': copy2cpu(val_loss)}

    def _get_data(self, split_name):

        dataset = get_dataset(self.config, split_name)

        return DataLoader(dataset,
                          batch_size=self.config.TRAIN.BATCH_SIZE,
                          shuffle=True if split_name == 'train' else False,
                          num_workers=self.config.TRAIN.NUM_WORKERS,
                          pin_memory=True)

    def train_dataloader(self):
        return self._get_data('train')
    
    def val_dataloader(self):
        return self._get_data('val')

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
                'frequency': 1
            },
        ]
        return [gen_optimizer], schedulers
    
    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['state_dict'] = self.denoiser.state_dict()

        