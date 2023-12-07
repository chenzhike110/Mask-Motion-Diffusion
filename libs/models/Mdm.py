import torch
from torch import nn 
from diffusers import DDIMScheduler

class MDM(nn.Module):
    """
    Latent Motion Diffusion
    """
    def __init__(self,
                 num_inference_timesteps=50,
                 vae:nn.Module = None) -> None:
        super().__init__()

        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            variance_type='fixed_small',
            clip_sample=False
        )
        self.num_inference_timesteps=num_inference_timesteps
    
    def forward(self, batch):
        texts = batch["text"]
        lengths = batch["length"]

        text_emb = self.text_encoder(texts)
        z = self._diffusion_reverse(text_emb, lengths)

        with torch.no_grad():
            pass
