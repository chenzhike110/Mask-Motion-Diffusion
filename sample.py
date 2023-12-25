import torch
import numpy as np
from dataclasses import dataclass
from libs.config import parse_args
from libs.get_model import get_model_with_config
from libs.data import HumanML3DDataModule
from transformers import AutoModel, AutoTokenizer

@dataclass
class blankDataset:
    text_zero_padding: np.array

def main():
    cfg = parse_args()

    text_encoder = AutoModel.from_pretrained('/home/czk119/Desktop/Motion Generation/motion-latent-diffusion/deps/clip-vit-base-patch16').cuda()
    tokenizer = AutoTokenizer.from_pretrained('/home/czk119/Desktop/Motion Generation/motion-latent-diffusion/deps/clip-vit-base-patch16')

    with torch.no_grad():
        text_inputs = tokenizer(
            ["", "a man waves his right hand"], 
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        caption = text_encoder.get_text_features(
            text_input_ids.to(text_encoder.device)
        ).cpu().numpy()

    datamodule = blankDataset(caption[0:1])

    model = get_model_with_config(cfg, datamodule)
    model.load_state_dict(torch.load("saved/epoch=519_val_loss=0.45.ckpt")['state_dict'])
    model = model.to(torch.device("cuda"))
    model.eval()

    

    with torch.no_grad():
        smpls = model.sample({
            "text":[caption[1:]], 
            "length":[50]
        }).cpu()
        smpls = HumanML3DDataModule(cfg).recover_motion(smpls)
        np.save('demo.npy', smpls.v.cpu().numpy())


if __name__ == "__main__":
    main()

