import random
from copygec.mytokenizer import PAD_IDX, enc
from copygec.utils import read_json, to_padded_tensor, unzip

def load_datasets():
	return read_json('./data/data.json')

def sentences_to_padded_tensor(sentences):
	return to_padded_tensor([enc(s) for s in sentences], PAD_IDX).T


class DataLoader:
	def __init__(self, xys, batch_size, device):
		self.xys = xys.copy()
		self.batch_size = batch_size
		self.device = device
	
	def __len__(self):
		return len(self.xys)

	def __iter__(self):
		self.iter = 0
		bs = self.batch_size
		random.shuffle(self.xys)
		self.xys.sort(key=lambda xy: len(xy[0]))
		s = len(self.xys)
		self.batches = [self.xys[i*bs:(i+1)*bs] for i in range(s//bs)]
		if s % bs != 0: self.batches.append(self.xys[s//bs*bs:])
		random.shuffle(self.batches)
		return self

	def __next__(self):
		if self.iter < len(self.batches):
			xs, ys = unzip(self.batches[self.iter])
			self.iter += 1
			return sentences_to_padded_tensor(xs).to(self.device), sentences_to_padded_tensor(ys).to(self.device)
		else:
			raise StopIteration
