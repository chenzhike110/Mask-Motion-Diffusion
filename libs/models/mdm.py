import torch
from torch import nn 
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoTokenizer

from ..body_model import BodyModel
from ..get_model import get_model
from .diffusion import create_diffusion
from .embedding import LabelEmbedder

class MDM(LightningModule):
    """
    Latent Motion Diffusion
    """
    def __init__(self,
                 config) -> None:
        super().__init__()

        self.config = config

        # condition on action class
        self.class_embedder = LabelEmbedder(config.MODEL.ACTION.num_classes, config.MODEL.hidden_size, config.MODEL.ACTION.class_dropout_prob)
        self.text_embedder = AutoModel.from_pretrained(config.MODEL.TEXT.ENCODER)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT.TOKENIZER)
        # y = self.y_embedder(y, self.training)    # (N, D)        
        self.num_inference_timesteps=config.MODEL.num_inference_timesteps
        self.vae = get_model(
            config.MODEL.VAE
        )
        self.vae.load_state_dict(torch.load(config.VAE.checkpoint))
        
        # freeze vae
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # freeze text model
        for param in self.text_embedder.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            self.bm = BodyModel(config.SMPL_PATH)

        self.diffusion = create_diffusion(timestep_respacing=config.MODEL.timestep_respacing)
        self.denoiser = get_model(config.MODEL.DENOISER)
        self.rootNet = nn.Sequential(
            
        )

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
    
    def forward(self, batch):
        # get input
        texts = batch["text"]
        lengths = batch["length"]
        
        
        
        return 

    def training_step(self, batch):
        # get input
        texts = batch["text"]
        lengths = batch["length"]
        joint_rot = batch['pose_body'].view(-1, 21*6) #todo: add joint number to config
        root_pos = batch['pose_root'].view(-1, 3)
        root_rot = batch['orien_root'].view(-1, 6)

        # get latent
        with torch.no_grad():
            q_z_sample, q_z = self.vae.encode(joint_rot)
        
        # diffusion
        x = torch.concatenate([root_pos, root_rot, q_z_sample], dim=-1)
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=joint_rot.device)
        model_kwargs = dict(y=y)
        loss_dict = self.diffusion.training_losses(self.denoiser, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        
        return
    
