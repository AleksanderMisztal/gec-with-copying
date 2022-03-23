import string
import torch
import json
import random

def zip_tensors(a, b):
  a = a.detach().numpy().tolist()
  b = b.detach().numpy().tolist()

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

def writelines(path, lines):
  with open(path, "w", encoding='utf-8') as f:
    f.write('\n'.join(lines))

def show_cuda_memory():
  MB = 10 ** 6
  t = torch.cuda.get_device_properties(0).total_memory//MB
  r = torch.cuda.memory_reserved(0)//MB
  a = torch.cuda.memory_allocated(0)//MB
  f = r-a
  print(f'Total: {t}, reserved: {r}, allocated: {a}, free: {f}')

chars = string.ascii_letters + string.digits
with open('./data/words.txt') as f: words = f.read().split()

def rand_char():
  return random.choice(chars)

def rand_word():
  return random.choice(words)

def noise(x):
  tokens = x.split()
  tok_changes = len(tokens)//7
  for i in random.sample(range(len(tokens)-3), tok_changes):
    if i >= len(tokens): continue
    t = list(tokens[i])
    a = 100*random.random()
    if a < 40:
      t[random.choice(range(len(t)))] = rand_char()
      tokens[i] = ''.join(t)
    elif a < 60:
      tokens.pop(i)
    elif a < 80:
      tokens.insert(i, rand_word())
    elif i > 0:
      tokens[i-1:i+1] = [tokens[i], tokens[i-1]]
  return ' '.join(tokens)


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