import torch
import numpy as np
from libs.config import parse_args
from libs.get_model import get_model_with_config, get_dataset

device = torch.device('cuda:2')

def main():
    cfg = parse_args()

    datamodule = get_dataset(cfg)
    model = get_model_with_config(cfg, datamodule=datamodule).to(device)

    




