import random
import torch
import json

def unzip(xys):
  xs = [x for x, y in xys]
  ys = [y for x, y in xys]
  return xs, ys

def to_padded_tensor(xss, pad_token):
	w = max(len(xs) for xs in xss)
	padded = [xs + [pad_token for _ in range(w - len(xs))] for xs in xss]
	return torch.tensor(padded)

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

def read_json(path):
  with open(path, 'r') as f:
    return json.load(f)