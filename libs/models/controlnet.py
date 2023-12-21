import torch
import torch.nn as nn

from .utils import conv_nd, zero_module

class controlNet(nn.Module):
    """
    ControlNet for 3D sequence data
    """
    def __init__(
        self,
        hint_channels,
        model_channels,
        dims=2,
    ):
        
        self.input_hint_block = nn.Sequential(
                conv_nd(dims, hint_channels, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
            )