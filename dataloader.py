import random
from utils import read_json, to_padded_tensor, unzip

DATA_PATH = './data/'
def load_datasets():
	train = read_json(DATA_PATH + 'train.json')
	val = read_json(DATA_PATH + 'val.json')
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

