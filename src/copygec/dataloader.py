import random
from .utils import read_json, to_padded_tensor, unzip

def load_datasets(path):
	train = read_json(path + 'train.json')
	val = read_json(path + 'val.json')
	return train, val

class DataLoader:
	def __init__(self, xys, batch_size, pad_token):
		self.xys = xys.copy()
		self.batch_size = batch_size
		self.pad_token = pad_token
	
	def __len__(self):
		return len(self.xys)

	def __iter__(self):
		self.iter = 0
		bs = self.batch_size
		random.shuffle(self.xys)
		self.xys.sort(key=lambda xy: len(xy[0]))
		self.batches = [self.xys[i*bs:(i+1)*bs] for i in range(len(self.xys)//bs)]
		random.shuffle(self.batches)
		return self

	def __next__(self):
		if self.iter < len(self.batches):
			xs, ys = unzip(self.batches[self.iter])
			self.iter += 1
			return to_padded_tensor(xs, self.pad_token).T, to_padded_tensor(ys, self.pad_token).T
		else:
			raise StopIteration
