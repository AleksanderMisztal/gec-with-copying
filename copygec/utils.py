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

def count_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_json(path):
  with open(path, 'r') as f:
    return json.load(f)

def writelines(path, lines):
  with open(path, "w") as f:
    f.write('\n'.join(lines))


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