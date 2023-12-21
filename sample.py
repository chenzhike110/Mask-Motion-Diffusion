import torch
import numpy as np
from libs.config import parse_args
from libs.get_model import get_model_with_config

def main():
    cfg = parse_args()
    model = get_model_with_config(cfg)
    model.denoiser.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT)['state_dict'])
    model = model.to(torch.device("cuda"))
    model.eval()
    with torch.no_grad():
        smpls = model.sample({
            "text":['a man waves his right hand'], 
            "length":[50]
        })
        np.save('demo.npy', smpls.v[:, :, [0,2,1]].cpu().numpy())


if __name__ == "__main__":
    main()

