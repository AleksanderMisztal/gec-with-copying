import torch
import torch.nn as nn

class RefTransformer(nn.Module):
  def __init__(self, vocab_s, emb_s=256):
      super(RefTransformer, self).__init__()

      self.embedding = nn.Embedding(vocab_s, emb_s)
      self.transformer = nn.Transformer(d_model=emb_s)

  def forward(self, x, tgt):
    x_emb = self.embedding(x)
    tgt_emb = self.embedding(tgt)
    y = self.transformer(x_emb, tgt_emb)
    
    return y
  
