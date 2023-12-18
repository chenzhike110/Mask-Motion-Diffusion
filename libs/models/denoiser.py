import torch
import torch.nn as nn
import numpy as np
from .embedding import TimestepEmbedder, build_position_encoding
from .transformer import BasicTransformerBlock, _get_clones, SequenceTransformer
from .cross_attention import TransformerEncoderLayer, TransformerEncoder

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
                 position_embedding='learned',
                 normalize_before=False,
                 activation="gelu"
    ):
        super().__init__()

        self.patch_size = patch_size
        self.output_size = input_size * 2 if learn_sigma else input_size

        self.x_embedder = nn.Linear(input_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = build_position_encoding(
                hidden_size, position_embedding=position_embedding)
        # self.pos_embed = nn.Parameter(torch.zeros(1, patch_size, hidden_size), requires_grad=False)

        # seqTransEncoderLayer = BasicTransformerBlock(dim=hidden_size, n_heads=num_heads, d_head=64, dropout=dropout, gated_ff=False)
        # self.seqTransEncoder = SequenceTransformer(seqTransEncoderLayer, depth)
        # seqTransEncoderLayer =  nn.TransformerEncoderLayer(d_model=hidden_size,
        #                                      nhead=num_heads,
        #                                      dim_feedforward=int(mlp_ratio*hidden_size),
        #                                      dropout=dropout,
        #                                      activation=activation)
        # self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
        #                                           num_layers=depth)
        
        seqTransEncoderLayer = TransformerEncoderLayer(
                    hidden_size,
                    num_heads,
                    int(mlp_ratio*hidden_size),
                    dropout,
                    activation,
                    normalize_before,
                )
        self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                depth)

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

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, y, attention_mask=None):
        x = self.x_embedder(x)
        t = self.t_embedder(t).unsqueeze(1)

        # c = t + y or c = (t, y)
        c = torch.cat((t, y), 1)
        c_maks = torch.ones((x.shape[0], 2), dtype=torch.bool).to(attention_mask.device)
        attention_mask = torch.cat((c_maks, attention_mask), 1)

        xseq = torch.cat((c, x), 1).permute(1, 0, 2) #(seq+1, bs, d)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=attention_mask)[c.shape[1]:, :].permute(1, 0, 2)
        output = self.outFinal(output)
        return output



    
