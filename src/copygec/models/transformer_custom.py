import torch
import torch.nn as nn
from src.copygec.models.common import PositionalEncoding

cpu = torch.device('cpu')

class Encoder(nn.Module):
  def __init__(self, d_model, num_layers, nhead=8, device=cpu):
    super(Encoder, self).__init__()
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, device=device)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

  def forward(self, input):
    out = self.encoder(input)
    return out


class Decoder(nn.Module):
  def __init__(self, d_model, num_layers, nhead=8, device=cpu):
    super(Decoder, self).__init__()
    decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, device=device)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
  
  def forward(self, tgt, memory):
    mask = torch.nn.Transformer.generate_square_subsequent_mask(None, tgt.shape[0])
    out = self.decoder(tgt, memory, tgt_mask=mask)
    return out


class Transformer(nn.Module):
  def __init__(self, vocab_s, d_model, num_layers, pad_idx: int, device=cpu):
    super(Transformer, self).__init__()
    self.pad_idx = pad_idx
    self.embed = nn.Embedding(vocab_s, d_model, device=device)
    self.pos_embed = PositionalEncoding(d_model, .1).to(device)
    self.encoder = Encoder(d_model, num_layers, device=device)
    self.decoder = Decoder(d_model, num_layers, device=device)
    self.generator = nn.Linear(d_model, vocab_s, device=device)
  
  def encode(self, src:torch.Tensor) -> torch.Tensor:
    emb = self.pos_embed(self.embed(src))
    return self.encoder(emb)
  
  def decode(self, tgt, memory):
    emb = self.pos_embed(self.embed(tgt))
    out = self.decoder(emb, memory)
    return out 

  def forward(self, src, tgt):
    memory = self.encode(src)
    y = self.decode(tgt, memory)
    y = self.generator(y)
    return y