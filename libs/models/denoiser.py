import torch
import torch.nn as nn
import numpy as np
from .embedding import TimestepEmbedder, get_1d_sincos_pos_embed_from_grid

class Denoiser(nn.Module):
    """
    Denoiser with transformer arch
    """
    def __init__(self,
                 input_size=64,
                 hidden_size=512,
                 patch_size=256,
                 depth=6,
                 num_heads=4,
                 mlp_ratio=2.0,
                 dropout=0.1,
                 learn_sigma=False,
                 activation="gelu"
    ):
        super().__init__()

        self.patch_size = patch_size
        self.output_size = input_size * 2 if learn_sigma else input_size

        self.x_embedder = nn.Linear(input_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_size, hidden_size), requires_grad=False)
        
        seqTransEncoderLayer =  nn.TransformerEncoderLayer(d_model=hidden_size,
                                             nhead=num_heads,
                                             dim_feedforward=int(mlp_ratio*hidden_size),
                                             dropout=dropout,
                                             activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=depth)
        
        self.outFinal = nn.Linear(hidden_size, self.output_size)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos embedding with cos:
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.patch_size, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, y, attention_mask=None):
        x = self.x_embedder(x) 
        x = x + self.pos_embed
        t = self.t_embedder(t).unsqueeze(1)
        c = t + y
        xseq = torch.cat((c, x), axis=1).permute(1,0,2) #(seq+1, bs, d)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=attention_mask)[1:]
        output = self.outFinal(output)
        return output.permute(1,0,2)



    
