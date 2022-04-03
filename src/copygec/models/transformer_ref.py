import torch
import torch.nn as nn
from copygec.models.pos_encoding import PositionalEncoding

class Transformer(nn.Module):
  def __init__(self, vocab_s, pad_idx, num_layers=1, d_model=512, device=None):
    super(Transformer, self).__init__()
    if device is None: device = torch.device('cpu')
    self.device = device
    self.pad_idx = pad_idx
    self.embedding = nn.Embedding(vocab_s, d_model).to(device)
    self.pos_embedding = PositionalEncoding(d_model, 0.1).to(device)
    self.transformer = nn.Transformer(d_model=d_model, dim_feedforward=1024, num_encoder_layers=num_layers, num_decoder_layers=num_layers).to(device)
    self.generator = nn.Linear(d_model, vocab_s).to(device)
    self.d_model = d_model

  def forward(self, src, tgt):
    memory = self.encode(src)
    y = self.decode(tgt, memory)
    return y
  
  def encode(self, src):
    src_padding_mask = (src == self.pad_idx).transpose(0, 1)
    emb = self.pos_embedding(self.embedding(src))
    return self.transformer.encoder(emb, src_key_padding_mask=src_padding_mask)

  def decode(self, tgt, memory):
    sz = tgt.shape[0]
    tgt_mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)
    tgt_padding_mask = (tgt == self.pad_idx).transpose(0, 1)
    emb = self.pos_embedding(self.embedding(tgt))
    y = self.transformer.decoder(emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
    y = self.generator(y)
    return y
  
  
