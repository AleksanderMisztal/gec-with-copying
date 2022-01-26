import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  def __init__(self,
              emb_size: int,
              dropout: float,
              maxlen: int = 5000):
    super(PositionalEncoding, self).__init__()
    den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
    pos = torch.arange(0, maxlen).reshape(maxlen, 1)
    pos_embed = torch.zeros((maxlen, emb_size))
    pos_embed[:, 0::2] = torch.sin(pos * den)
    pos_embed[:, 1::2] = torch.cos(pos * den)
    pos_embed = pos_embed.unsqueeze(-2)

    self.dropout = nn.Dropout(dropout)
    self.register_buffer('pos_embed', pos_embed)

  def forward(self, token_embedding) -> torch.Tensor:
    return self.dropout(token_embedding + self.pos_embed[:token_embedding.size(0), :])
