import torch
from torch import nn
import torch.nn.functional as F
from libs.models.embedding import PositionEmbeddingSine1D, TimestepEmbedder

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x
    
class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, ff_size, dropout, num_layers, latent_dim, input_size, output_size, **kargs):
        super().__init__()
        seqTransEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = torch.nn.TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=num_layers)
        
        self.position_enc = PositionEmbeddingSine1D(latent_dim)
        self.t_embedder = TimestepEmbedder(latent_dim)
        self.x_embedder = InputProcess(input_feats=input_size, latent_dim=latent_dim)
        self.output_process = OutputProcess(out_feats=output_size, latent_dim=latent_dim)

        self.output_size = output_size
        
    def forward(self, x, t, cond, mask=None):
        x = self.x_embedder(x)
        x = x + self.position_enc(x)

        t = self.t_embedder(t).unsqueeze(1)
        cond = torch.cat((cond, t), 1).permute((1, 0, 2))

        xseq = torch.cat([cond, x], dim=0) #(seqlen+1, b, latent_dim)

        if mask:
            mask = torch.cat([torch.zeros_like(mask[:, 0:2]), mask], dim=1) #(b, seqlen+1)
        # print(xseq.shape, padding_mask.shape)

        # print(padding_mask.shape, xseq.shape)

        output = self.seqTransEncoder(xseq, src_key_padding_mask=mask)[cond.shape[0]:] #(seqlen, b, e)
        output = self.output_process(output).permute((1, 0, 2))
        return output