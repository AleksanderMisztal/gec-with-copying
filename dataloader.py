import random
import torch

def to_padded_tensor(xss, pad_token):
	w = max(len(xs) for xs in xss)
	padded = [xs + [pad_token for _ in range(w - len(xs))] for xs in xss]
	return torch.tensor(padded)

class DataLoader:
	def __init__(self, xys, batch_size, pad_token):
		self.xys = xys
		self.batch_size = batch_size
		self.pad_token = pad_token
		
	def __iter__(self):
		self.iter = 0
		random.shuffle(self.xys)
		return self

	def __next__(self):
		if self.iter + self.batch_size <= len(self.xys):
			batch = self.xys[self.iter:self.iter + self.batch_size]
			xs = [x for x, y in batch]
			ys = [y for x, y in batch]
			self.iter += self.batch_size
			return to_padded_tensor(xs, self.pad_token), to_padded_tensor(ys, self.pad_token)
		else:
			raise StopIteration


def load_gec_data(short=False):
	orig1 = open('./data/conll.orig.txt').readlines()
	corr1 = open('./data/conll.corr.txt').readlines()

	orig2 = open('./data/nucle.orig.txt').readlines()
	corr2 = open('./data/nucle.corr.txt').readlines()

	orig = orig1 + orig2
	corr = corr1 + corr2

	orig = [s.strip() for s in orig]
	corr = [s.strip() for s in corr]

	xys = list(zip(orig, corr))
	xys = [(x, y) for x, y in xys if x != y]

	if short:
		xys = [(x,y) for x,y in xys if len(x) < 50 and len(y) < 50]

	return xys