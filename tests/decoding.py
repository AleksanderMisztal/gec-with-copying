import random
import torch
from copygec.dataloader import load_datasets
from copygec.decoding import beam_search_decode, greedy_decode
from copygec.model import RefTransformer
from copygec.mytokenizer import tokenizer

from copygec.utils import writelines


MODEL_LOAD_NAME = 'transformer/model_eos'
model = RefTransformer(tokenizer.get_vocab_size())
model.load_state_dict(torch.load('./models/' + MODEL_LOAD_NAME + '.pt'))

_, xys_val = load_datasets('./data/')

for x,y in random.sample(xys_val,1):
  print(x)
  print(y)
  pred, logprop = greedy_decode(model, x)
  print(round(pred,3), logprop)
  preds = beam_search_decode(model, x)
  for lp, pr in preds:
    print(round(lp,3), pr)
  print()

N_SENTENCES = 24
USE_ALL = True
if USE_ALL:
  sample = xys_val
else:
  if N_SENTENCES > len(xys_val):
    print(f"Specified number = {N_SENTENCES} > {len(xys_val)} = len of val set")
    exit()
  sample = random.sample(xys_val, N_SENTENCES)
preds = []
for i,(x,y) in enumerate(sample):
  print(f"{i+1}/{len(sample)}", end='\r')
  prob, pred = beam_search_decode(model, x, n_results=1)[0]
  preds.append([x,y,pred])

writelines("../out/orig.txt", [x for x,y,p in preds])
writelines("../out/corr.txt", [y for x,y,p in preds])
writelines("../out/pred.txt", [p for x,y,p in preds])
