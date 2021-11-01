import random
from utils import to_padded_tensor, unzip

orig1 = open('./data/conll.orig.txt').readlines()
corr1 = open('./data/conll.corr.txt').readlines()

orig2 = open('./data/nucle.orig.txt').readlines()
corr2 = open('./data/nucle.corr.txt').readlines()

orig = [s.strip() for s in orig1 + orig2]
corr = [s.strip() for s in corr1 + corr2]
xys = list(zip(orig, corr))
xys = [(x, y) for x, y in xys if x != y]

orig, corr = unzip(xys)

def get_orig_and_corr(): return orig, corr
def get_orig_corr_pairs(lim=-1): return xys if lim == -1 else [(x,y) for x, y in xys if len(x) < lim and len(y) < lim]

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
			xs, ys = unzip(batch)
			self.iter += self.batch_size
			return to_padded_tensor(xs, self.pad_token).T, to_padded_tensor(ys, self.pad_token).T
		else:
			raise StopIteration

