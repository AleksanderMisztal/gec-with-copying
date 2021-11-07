import torch
import random
from dataloader import load_datasets
from model import RefTransformer
from mytokenizer import BOS_IDX, EOS_IDX, tokenizer
from utils import writelines


model = RefTransformer(tokenizer.get_vocab_size())

MODEL_LOAD_NAME = 'model_eos'
model.load_state_dict(torch.load('./models/' + MODEL_LOAD_NAME + '.pt'))

def greedy_decode(model, src, max_len):
  ys = torch.tensor([[BOS_IDX]])
  memory = model.encode(src)
  for i in range(max_len):
    out = model.decode(ys, memory)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()
    ys = torch.cat([ys, torch.tensor([[next_word]])], dim=0)
    if next_word == EOS_IDX or next_word==13: break
  return ys


def correct_sentence(sentence):
  src = tokenizer.encode(sentence).ids
  idxs = greedy_decode(model, torch.tensor([src]).T, 100)
  pred = tokenizer.decode(idxs.T[0].tolist()).strip()
  return pred


_, xys_val = load_datasets()
N_SENTENCES = 24
USE_ALL = False
if USE_ALL:
  sample = xys_val
else:
  if N_SENTENCES > len(xys_val):
    print(f"Specified number = {N_SENTENCES} > {len(xys_val)} = len of val set")
    exit()
  sample = random.sample(xys_val, N_SENTENCES)
preds = []
for i,(x,y) in enumerate(sample):
  print(f"{i+1}/{N_SENTENCES}", end='\r')
  pred = correct_sentence(x)
  preds.append([x,y,pred])

writelines("./out/orig.txt", [x for x,y,p in preds])
writelines("./out/corr.txt", [y for x,y,p in preds])
writelines("./out/pred.txt", [p for x,y,p in preds])
