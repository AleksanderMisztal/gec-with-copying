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
    pos_embedding = torch.zeros((maxlen, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    pos_embedding = pos_embedding.unsqueeze(-2)

    self.dropout = nn.Dropout(dropout)
    self.register_buffer('pos_embedding', pos_embedding)

  def forward(self, token_embedding):
    return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Encoder(nn.Module):
  def __init__(self, vocab_s, emb_s):
    super(Encoder, self).__init__()
    self.embed = nn.Embedding(vocab_s, emb_s)
    self.pos_embed = PositionalEncoding(emb_s, .1)
    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

  def forward(self, input):
    emb = self.pos_embed(self.embed(input))
    out = self.encoder(emb)
    return out

class EncoderClassifier(nn.Module):
  def __init__(self, vocab_s, emb_s, n_classes):
    super(EncoderClassifier, self).__init__()
    self.encoder = Encoder(vocab_s, emb_s)
    self.predict = nn.Linear(emb_s, n_classes)

  def forward(self, x):
    encoded = self.encoder(x)
    out = self.predict(encoded)
    return out

class RefTransformer(nn.Module):
  def __init__(self, vocab_s, emb_s=256):
    super(RefTransformer, self).__init__()

    self.embedding = nn.Embedding(vocab_s, emb_s)
    self.pos_embedding = PositionalEncoding(emb_s, 0.1)
    self.transformer = nn.Transformer(d_model=256, dim_feedforward=1024)
    self.generator = nn.Linear(emb_s, vocab_s)

  def forward(self, x, tgt):
    x_emb = self.pos_embedding(self.embedding(x))
    tgt_emb = self.pos_embedding(self.embedding(tgt))
    y = self.transformer(x_emb, tgt_emb)
    y = self.generator(y)
    
    return y
  
  # def encode(self, src, src_mask):
  #   return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

  # def decode(self, tgt, memory, tgt_mask):
  #   return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
  
