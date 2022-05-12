import copy
import torch
import json
import random

def zip_tensors(a, b):
  a = a.cpu().detach().numpy().tolist()
  b = b.cpu().detach().numpy().tolist()

  return [list(zip(ai, bi)) for ai, bi in zip(a,b)]


def unzip(xys):
  xs = [x for x, y in xys]
  ys = [y for x, y in xys]
  return xs, ys

def to_padded_tensor(xss, pad_token):
	w = max(len(xs) for xs in xss)
	padded = [xs + [pad_token for _ in range(w - len(xs))] for xs in xss]
	return torch.tensor(padded)

def count_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)

def write_json(path, data):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

def writelines(path, lines):
  with open(path, "w", encoding='utf-8') as f:
    f.write('\n'.join(lines))

def readlines(path):
  with open(path, "r", encoding='utf-8') as f:
    return f.readlines()

def show_cuda_memory():
  MB = 10 ** 6
  t = torch.cuda.get_device_properties(0).total_memory//MB
  r = torch.cuda.memory_reserved(0)//MB
  a = torch.cuda.memory_allocated(0)//MB
  f = r-a
  print(f'Total: {t}, reserved: {r}, allocated: {a}, free: {f}')

def mask(tokens, vocab_s=30_000, mask_idx=3):
  tokens = copy.copy(tokens)
  loss_mask = [0 for _ in range(len(tokens))]
  for i in range(1,len(tokens)-1):
    a = random.random()*100
    if a > 16: continue
    elif a > 8: tokens[i] = mask_idx
    elif a > 4: tokens[i] = random.randint(4, vocab_s-1)
    loss_mask[i] = 1
  return tokens#, loss_mask


from heapq import heapify, heappush, heappushpop

class MaxHeap():
  def __init__(self, top_n):
    self.h = []
    self.length = top_n
    heapify( self.h)
      
  def add(self, element):
    if len(self.h) < self.length:
      heappush(self.h, element)
    else:
      heappushpop(self.h, element)
  
  def addAll(self, elems):
    for elem in elems:
      self.add(elem)

  def getTop(self):
    return sorted(self.h, reverse=True)

