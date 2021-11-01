import random
import torch

def unzip(xys):
  xs = [x for x, y in xys]
  ys = [y for x, y in xys]
  return xs, ys

def create_mask(src, tgt_in, pad_idx):
  t_len = tgt_in.shape[1]

  tgt_mask = torch.triu(torch.ones(t_len, t_len), diagonal=1) * float('-inf')

  src_padding_mask = (src == pad_idx) 
  tgt_padding_mask = (tgt_in == pad_idx)

  return tgt_mask, src_padding_mask, tgt_padding_mask

def to_padded_tensor(xss, pad_token):
	w = max(len(xs) for xs in xss)
	padded = [xs + [pad_token for _ in range(w - len(xs))] for xs in xss]
	return torch.tensor(padded)

def get_tf_predictions(model, sentence_pairs, pad):
  xs, ys = unzip(sentence_pairs)
  src, tgt = to_padded_tensor(xs, pad).T, to_padded_tensor(ys, pad).T
  out = model(src, tgt[:-1, :])
  words = torch.argmax(out, dim=2)
  idxs = words.T.tolist()
  return idxs

def noise(orig: 'list[int]', vocab_s: int):
  x = orig.copy()
  y = [0 for _ in x]
  i = 0
  while i < len(x)-1:
    a = random.random()
    if a < .15:
      y[i] = y[i+1] = 1
      x[i], x[i+1] = x[i+1], x[i]
      i+=1
    elif a < .3:
      x[i] = random.randint(0, vocab_s-1)
      y[i] = 2
    i+=1
  return x, y

def count_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)



