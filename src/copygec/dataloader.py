import random
from copygec.mytokenizer import PAD_IDX, enc
from copygec.utils import read_json, to_padded_tensor, unzip
from itertools import islice

def load_datasets(name='data', dummy=False):
	data = read_json(f'./data/{name}.json')
	if not dummy: return data

	data['train'] = data['train'][:10]
	data['dev'] = data['dev'][:10]
	data['test'] = data['test'][:10]
	return data

def load_pretraining_sentences(filename, chunk_size=100_000):
	with open(filename, 'r', encoding='utf-8') as f:
		while True:
			next_n_lines = list(islice(f, chunk_size))
			if next_n_lines: yield next_n_lines
			else: break

def sentences_to_padded_tensor(sentences):
	return to_padded_tensor([enc(s) for s in sentences], PAD_IDX).T


class DataLoader:
	def __init__(self, xys, batch_size, device, preprocess=None):
		self.xys = xys
		self.batch_size = batch_size
		self.device = device
		self.preprocess = preprocess
	
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
			if self.preprocess is not None: xs = [self.preprocess(x) for x in xs]
			self.iter += 1
			return sentences_to_padded_tensor(xs).to(self.device), sentences_to_padded_tensor(ys).to(self.device)
		else:
			raise StopIteration
