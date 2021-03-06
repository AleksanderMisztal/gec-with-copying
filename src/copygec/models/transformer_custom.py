import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from copygec.models.pos_encoding import PositionalEncoding

def make_model(vocab_s, pad_idx, copy=False, num_layers=1, d_model=512, d_ff=1024, h=8, dropout=0.1, device=None):
  if device is None: device = torch.device('cpu')
  
  attn = lambda: MultiHeadedAttention(h, d_model).to(device)
  ff = lambda: PositionwiseFeedForward(d_model, d_ff, dropout)
  embed = nn.Sequential(
    Embeddings(vocab_s, d_model),
    PositionalEncoding(d_model, dropout)
  )
  embedding_layer = embed[0].emb

  model = CopyEncoderDecoder(
    Encoder(EncoderLayer(d_model, attn(), ff(), dropout), num_layers),
    Decoder(DecoderLayer(d_model, attn(), attn(), ff(), dropout), num_layers),
    embed,
    embed,
    CopyGenerator(d_model, vocab_s, embedding_layer, is_copying=copy),
    pad_idx,
    device
  )

  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  
  return model.to(device)


class CopyEncoderDecoder(nn.Module):
  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pad_idx, device):
    super(CopyEncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator
    self.pad_idx = pad_idx
    self.d_model = encoder.layers[0].d_model
    self.device = device
      
  def forward(self, src, tgt):
    memory = self.encode(src)
    y = self.decode(tgt, memory, src)
    return y
  
  def encode(self, src):
    src_padding_mask = (src == self.pad_idx).transpose(0, 1)
    emb = self.src_embed(src)
    return self.encoder(emb)
  
  def decode_only(self, tgt, memory):
    sz = tgt.shape[0]
    tgt_subsequent_mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)
    tgt_padding_mask = (tgt == self.pad_idx).transpose(0, 1).to(self.device)
    tgt_emb = self.tgt_embed(tgt)
    return self.decoder(tgt_emb, memory, tgt_subsequent_mask)
  
  def decode(self, tgt, memory, src):
    htgt = self.decode_only(tgt, memory)
    y = self.generator(htgt, memory, src)
    return y


class CopyGenerator(nn.Module):
  def __init__(self, d_model, vocab_s, emb_layer, is_copying=True):
    super(CopyGenerator, self).__init__()
    self.vocab_s = vocab_s
    self.attn = MultiHeadedAttention(1, d_model, dropout=0)
    self.copy_prob_lin = nn.Linear(d_model, 1)
    #self.generator = nn.Linear(d_model, vocab_s)
    self.emb_layer = emb_layer
    self.copy_data = None
    self.is_copying = is_copying

  def forward(self, htgt, hsrc, src):
    gen_score = torch.matmul(htgt, self.emb_layer.weight.T) #self.generator(htgt)
    if not self.is_copying: return gen_score
    # Nt, bs, d_model ; bs, Nt, Ns
    scores, attns = self.attn(htgt, hsrc, hsrc, return_attns=True)
    copy_p = pos_probs_to_idx_probs(src, attns, self.vocab_s)
    # a_copy.shape = Nt, bs, 1
    a_copy = torch.sigmoid(self.copy_prob_lin(scores))
    # Nt, bs, vocab_s = Nt,bs,1 ; Nt,bs,vocab_s ; Nt,bs,vocab_s
    gen_p = torch.softmax(gen_score, dim=2)
    self.copy_data = {'a': a_copy.detach(), 'copy': copy_p.detach(), 'gen': gen_p.detach()}
    return torch.log(a_copy * copy_p + (1.-a_copy) * gen_p)


def pos_probs_to_idx_probs(src, scores, vocab_s):
  oh = F.one_hot(src, vocab_s) * 1.0
  return torch.bmm(scores, oh.transpose(0,1)).transpose(0,1)
  


class Encoder(nn.Module):
  def __init__(self, layer, num_layers):
    super(Encoder, self).__init__()
    self.layers = cloned_layer(layer, num_layers)
    self.norm = LayerNorm(layer.d_model)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return self.norm(x)


class EncoderLayer(nn.Module):
  def __init__(self, d_model, self_attn, feed_forward, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn = self_attn # May be necessary for .to(device) to work properly
    self_attn_lmbd = lambda x: self_attn(x,x,x)
    self.self_attn_sl = ResidualSublayer(self_attn_lmbd, d_model, dropout)
    self.ff_sl = ResidualSublayer(feed_forward, d_model, dropout)
    self.d_model = d_model

  def forward(self, x):
    x = self.self_attn_sl(x)
    return self.ff_sl(x)


class Decoder(nn.Module):
  def __init__(self, layer, N):
    super(Decoder, self).__init__()
    self.layers = cloned_layer(layer, N)
    self.norm = LayerNorm(layer.d_model)
      
  def forward(self, x, memory, tgt_subsequent_mask):
    for layer in self.layers:
      x = layer(x, memory, tgt_subsequent_mask)
    return self.norm(x)


class DecoderLayer(nn.Module):
  def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
    super(DecoderLayer, self).__init__()
    self.self_attn = self_attn
    self_attn_lmbd = lambda x, mask: self_attn(x,x,x,mask)
    self.self_attn_sl = ResidualSublayer(self_attn_lmbd, d_model, dropout)
    self.src_attn_sl = ResidualSublayer(src_attn, d_model, dropout)
    self.ff_sl = ResidualSublayer(feed_forward, d_model, dropout)
    self.d_model = d_model

  def forward(self, x, memory, tgt_subsequent_mask):
    x = self.self_attn_sl(x, tgt_subsequent_mask)
    x = self.src_attn_sl(x, memory, memory)
    return self.ff_sl(x)


class ResidualSublayer(nn.Module):
  def __init__(self, sublayer, d_model, dropout):
    super(ResidualSublayer, self).__init__()
    self.sublayer = sublayer
    self.norm = LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, *args):
    return x + self.dropout(self.sublayer(self.norm(x), *args))


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
      
  def forward(self, q, k, v, subsequent_mask=None, return_attns=False):
    mask = subsequent_mask
    assert mask is None or mask.shape[0] == mask.shape[1]
    (Nt, bs, d_model) = q.shape
    (Ns, _, _) = k.shape
    # (Nt, bs, d) -> (bs*h, Nt, d)
    q = self.q_proj(q).view(Nt, bs * self.heads, self.d_vk).transpose(0,1)
    k = self.q_proj(k).view(Ns, bs * self.heads, self.d_vk).transpose(0,1)
    v = self.q_proj(v).view(Ns, bs * self.heads, self.d_vk).transpose(0,1)

    # attn shape: bs*h, Nt, Ns
    x, attns = sdp_attention(q, k, v, mask, self.dropout)
    # (bs*h, Nt, E) -> (Nt, bs, h*E)
    x = x.transpose(0,1).contiguous().view(Nt, bs, self.heads * self.d_vk)
    scores = self.final_proj(x)
    if return_attns: return scores, attns
    return scores


def sdp_attention(q, k, v, attn_mask, dropout):
  bs, Nt, E = q.shape
  q = q / math.sqrt(E)
  # (bs, Nt, E) x (bs, E, Ns) -> (bs, Nt, Ns)
  attn = torch.bmm(q, k.transpose(1, 2))
  if attn_mask is not None:
    attn += attn_mask
  attn = F.softmax(attn, dim=-1)
  if dropout is not None:
    attn = dropout(attn)
  # (bs, Nt, Ns) x (bs, Ns, E) -> (bs, Nt, E)
  output = torch.bmm(attn, v)
  return output, attn


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
    self.emb = nn.Embedding(vocab_s, d_model)
    self.d_model = d_model

  def forward(self, x):
    return self.emb(x) * math.sqrt(self.d_model)


def cloned_layer(module, n):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])