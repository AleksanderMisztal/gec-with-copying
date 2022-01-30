import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm # ! Annotated transformer has a custom implementation, idk if important
from copygec.models.common import PositionalEncoding


def make_model(vocab_s, num_layers=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
  attn = lambda: MultiHeadedAttention(h, d_model)
  ff = lambda: PositionwiseFeedForward(d_model, d_ff, dropout)
  embed = nn.Sequential(
    Embeddings(vocab_s, d_model),
    PositionalEncoding(d_model, dropout)
  )

  model = EncoderDecoder(
    Encoder(EncoderLayer(d_model, attn(), ff(), dropout), num_layers),
    Decoder(DecoderLayer(d_model, attn(), attn(), ff(), dropout), num_layers),
    embed,
    embed,
    Generator(d_model, vocab_s)
  )

  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform(p)
  
  return model


class EncoderDecoder(nn.Module):
  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator
      
  def forward(self, src, tgt, src_mask, tgt_mask):
    memory = self.encode(src, src_mask)
    return self.decode(memory, src_mask, tgt, tgt_mask)
  
  def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)
  
  def decode(self, memory, src_mask, tgt, tgt_mask):
    return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
  def __init__(self, d_model, num_layers, nhead=8):
    super(Encoder, self).__init__()
    encoder_layer = EncoderLayer(d_model, nhead)
    self.layers = cloned_layer(encoder_layer, num_layers)
    self.norm = LayerNorm(encoder_layer.size)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)


class EncoderLayer(nn.Module):
  def __init__(self, d_model, self_attn, feed_forward, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn_sl = ResidualSublayer(self_attn, d_model, dropout)
    self.ff_sl = ResidualSublayer(feed_forward, d_model, dropout)

  def forward(self, x, mask):
    x = self.self_attn_sl(x, x, x, mask)
    return self.ff_sl(x)


class Decoder(nn.Module):
  def __init__(self, layer, N):
    super(Decoder, self).__init__()
    self.layers = cloned_layer(layer, N)
    self.norm = LayerNorm(layer.size)
      
  def forward(self, x, memory, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, memory, src_mask, tgt_mask)
    return self.norm(x)


class DecoderLayer(nn.Module):
  def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
    super(DecoderLayer, self).__init__()
    self_attn = lambda x, mask: self_attn(x,x,x,mask)
    self.self_attn_sl = ResidualSublayer(self_attn, d_model, dropout)
    self.src_attn_sl = ResidualSublayer(src_attn, d_model, dropout)
    self.ff_sl = ResidualSublayer(feed_forward, d_model, dropout)

  def forward(self, x, memory, src_mask, tgt_mask):
    x = self.self_attn_sl(x, x, x, tgt_mask)
    x = self.src_attn_sl(x, memory, memory, src_mask)
    return self.ff_sl(x)


class ResidualSublayer(nn.Module):
  def __init__(self, sublayer, d_model, dropout):
    super(ResidualSublayer, self).__init__()
    self.sublayer = sublayer
    self.norm = LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, *args):
    return self.norm(x + self.dropout(self.sublayer(x, *args)))


class Generator(nn.Module):
  def __init__(self, d_model, vocab):
    super(Generator, self).__init__()
    self.proj = nn.Linear(d_model, vocab)
    self.log_softmax = nn.LogSoftmax(dim=-1)

  def forward(self, x):
    return self.log_softmax(self.proj(x))


def attention(query, key, value, mask=None, dropout=None):
  d_k = query.size(-1)
  scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = F.softmax(scores, dim = -1)
  if dropout is not None:
    p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
  def __init__(self, heads, d_model, dropout=0.1):
    super(MultiHeadedAttention, self).__init__()
    assert d_model % heads == 0
    
    self.d_vk = d_model // heads
    self.heads = heads
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.final_proj = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(p=dropout)
    
    self.attn = None
      
  def forward(self, query, key, value, mask=None):
    if mask is not None: mask = mask.unsqueeze(1)
    nbatches = query.size(0)
    
    query = self.q_proj(query).view(nbatches, -1, self.heads, self.d_vk).transpose(1, 2)
    key   = self.k_proj(  key).view(nbatches, -1, self.heads, self.d_vk).transpose(1, 2)
    value = self.v_proj(query).view(nbatches, -1, self.heads, self.d_vk).transpose(1, 2)
    
    x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
    x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.heads * self.d_vk)
    return self.final_proj(x)


class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.linear2(self.dropout(F.relu(self.linear1(x))))


class Embeddings(nn.Module):
  def __init__(self, vocab_s, d_model):
    super(Embeddings, self).__init__()
    self.lut = nn.Embedding(vocab_s, d_model)
    self.d_model = d_model

  def forward(self, x):
    return self.lut(x) * math.sqrt(self.d_model)


def cloned_layer(module, n):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])