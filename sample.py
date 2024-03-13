import clip
import torch
import numpy as np

from libs.smplx import SMPLHLayer
from libs.tools.config import parse_args
from libs.tools.parse import get_model

device = torch.device("cpu")
clip_model, clip_preprocess = clip.load('ViT-B/32', device=device, jit=False)
clip.model.convert_weights(clip_model) 
clip_model.eval()

def text_sample(model, length: list, texts: list):
    
    feat_clip_text = [clip_model.encode_text(clip.tokenize(texts, truncate=True).to(device)).float().cpu()]

    sample = model.sample({"text": feat_clip_text, "length": length})
    sample = torch.from_numpy(np.load('data/HumanML/joints/M000000.npy')).to(device)
    sample = model.recover_motion(sample)
    return sample

def trajectory_sample(model, trajectory: list, texts: list):

    trajectory = torch.tensor(trajectory, dtype=torch.float32).to(device)
    feat_clip_text = [clip_model.encode_text(clip.tokenize(texts, truncate=True)).float().cpu()]


if __name__ == "__main__":
    cfg = parse_args()
    cfg.load_checkpoint = True
    cfg.checkpoint = 'saved/motion_diffusion/checkpoints/epoch=4299_val_loss=1.32.ckpt'
    model = get_model(cfg)
    model.body_model = SMPLHLayer(model_path=cfg.smpl_path)
    model = model.to(device)

    length = [50]
    texts = ['a person is walking in place at a slow pace']
    sample = text_sample(model, texts=texts, length=length)
    np.save('./demos/walk_in_place.npy', sample.vertices.cpu().numpy())