import torch
import random
from torch.utils.data import DataLoader
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoTokenizer

from ..body_model import BodyModel
from ..get_model import get_model, get_dataset
from .diffusion import create_diffusion
from .embedding import LabelEmbedder
from .transform import recover_root_rot_pos, matrot2aa
from .tgm_conversion import quaternion_to_angle_axis, rotation_6d_to_matrix
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

        if not hasattr(config.TRAIN, "SEED") or config.TRAIN.SEED == -1:
            config.TRAIN.SEED = random.randint(0, 25536)
        print("set seed: ", config.TRAIN.SEED)
        make_deterministic(config.TRAIN.SEED)

        # condition on action class
        self.class_embedder = LabelEmbedder(config.MODEL.ACTION.num_classes, config.MODEL.DENOISER.args.hidden_size, config.MODEL.ACTION.class_dropout_prob)
        
        # condition on text
        self.text_embedder = AutoModel.from_pretrained(config.MODEL.TEXT.ENCODER)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT.TOKENIZER)

        self.num_inference_timesteps=config.MODEL.DENOISER.num_inference_timesteps
        self.vae = get_model(
            config.MODEL.VAE
        )
        
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
            learn_sigma=config.MODEL.DENOISER.args.learn_sigma
        )
        self.denoiser = get_model(config.MODEL.DENOISER)

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
        text_embeddings = text_embeddings.unsqueeze(1)
        return text_embeddings
    
    def generate_sequence_mask(self, length):
        mask = torch.arange(self.denoiser.patch_size+1).expand(len(length), self.denoiser.patch_size+1).to(length.device) < (length.unsqueeze(1) + 1)
        return mask
    
    def forward(self, batch):
        # get input
        texts = batch["text"]
        lengths = batch["length"]
        
        return 
    
    def _compute_loss(self, dorig, drec):
        """
        compute all loss with sample
        origin: root 4 + body 6*joint_num
        result: root 4 + body 3*joint_num
        """
        l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
        # motion reconstraction
        with torch.no_grad():
            r_rot_quat, r_pos = recover_root_rot_pos(dorig[:, :4])
            root_orient = quaternion_to_angle_axis(r_rot_quat).view(-1, 3)
            body_orient = matrot2aa(rotation_6d_to_matrix(dorig[:, 4:].reshape(-1, 6))).reshape(-1, (self.joints_num-1)*3)
            bm_orig = self.bm(pose_body=body_orient, root_orient=root_orient, trans=r_pos)

        r_rot_quat, r_pos = recover_root_rot_pos(drec[:, :4])
        root_orient = quaternion_to_angle_axis(r_rot_quat).view(-1, 3)
        body_orient = drec[:, 4:]
        bm_rec = self.bm(pose_body=body_orient, root_orient=root_orient, trans=r_pos)

        v2v = l1_loss(bm_rec.v, bm_orig.v)
        j2j = l1_loss(bm_rec.Jtr, bm_orig.Jtr)
        return {"v2v": v2v, "j2j": j2j}


    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # get input
        texts = batch["text"]
        # uncondition
        texts = [
            "" if random.random() < self.config.TRAIN.guidance_uncodp else i
            for i in texts
        ]
        lengths = batch["length"]
        joint_rot = batch['pose_body'].view(-1, (self.joints_num-1)*6)
        root_pos = batch['pose_root']

        # get latent
        q_z_sample, q_z = self.vae.encode(joint_rot)
        q_z_sample = q_z_sample.view(len(texts), -1, self.vae.latentD)
        
        text_embedding = self.get_text_embedding(texts)

        attention_mask = self.generate_sequence_mask(lengths)
        
        # diffusion
        x = torch.concatenate([root_pos, q_z_sample], dim=-1)
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=joint_rot.device)
        model_kwargs = dict(y=text_embedding, attention_mask=~attention_mask)
        loss_dict = self.diffusion.training_losses(self.denoiser, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        texts = batch["text"]
        lengths = batch["length"]
        pose_root_ori = batch['pose_root'] * batch['std'][0, :4] + batch['mean'][0, :4]
        d_orig = torch.concatenate([pose_root_ori, batch["pose_body"]], dim=-1)
        
        text_embedding = self.get_text_embedding(texts)
        attention_mask = self.generate_sequence_mask(lengths)
        
        # diffusion
        model_kwargs = dict(y=text_embedding, attention_mask=~attention_mask)
        z = self.diffusion.p_sample_loop(
            self.denoiser, 
            (len(texts), self.denoiser.patch_size, self.denoiser.output_size),
            model_kwargs=model_kwargs,
        )

        # decode
        z_body = z[:, :, 4:].view(-1, self.vae.latentD)
        pose_body_rec = self.vae.decode(z_body)['pose_body'].contiguous().view(len(texts), -1, (self.joints_num-1)*3)
        pose_root_rec = z[:, :, :4] * batch['std'][0, :4] + batch['mean'][0, :4]
        d_rec = torch.concatenate([pose_root_rec, pose_body_rec], dim=-1)

        # mask select
        d_orig = d_orig.view(-1, d_orig.shape[-1])
        d_rec = d_rec.view(-1, d_rec.shape[-1])
        attention_mask = attention_mask[:, 1:].flatten()
        
        d_orig = d_orig[attention_mask, :]
        d_rec = d_rec[attention_mask, :]
        assert d_orig.shape[0] == lengths.sum()
        rec_loss = self._compute_loss(d_orig, d_rec)

        for k, v in rec_loss.items():
            self.log(k, v, prog_bar=True, logger=True)

        val_loss = torch.stack(list(rec_loss.values())).sum()

        self.log("val_loss: ", val_loss, prog_bar=True, logger=True)
        return copy2cpu(val_loss)

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
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.denoiser.state_dict()

        