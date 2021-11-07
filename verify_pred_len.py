import torch
import random
from dataloader import load_datasets
from model import RefTransformer
from mytokenizer import BOS_IDX, EOS_IDX, tokenizer
from utils import MaxHeap, writelines

def greedy_decode(model, sentence, max_len=100):
  src = tokenizer.encode(sentence).ids
  src = torch.tensor([src]).T
  memory = model.encode(src)
  ys = torch.tensor([[BOS_IDX]])
  for i in range(max_len):
    out = model.decode(ys, memory)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()
    ys = torch.cat([ys, torch.tensor([[next_word]])], dim=0)
    if next_word == EOS_IDX or next_word==13: break
  
  pred = tokenizer.decode(ys.T[0].tolist()).strip()
  return pred

def beam_search_decode(model, sentence, n_beams=12, n_results=3):
  src = tokenizer.encode(sentence).ids
  src = torch.tensor([src]).T
  memory = model.encode(src)
  top_sentences = MaxHeap(n_results)
  beams = [(0,torch.tensor([[BOS_IDX]])) for i in range(n_beams)]
  max_len = max(10, int(1.5*len(src)))
  for i in range(max_len):
    h = MaxHeap(n_beams)
    for logp,ys in beams:
      out = model.decode(ys, memory)
      out = out.transpose(0, 1)
      logprobs = torch.nn.LogSoftmax(dim=1)(model.generator(out[:, -1]))
      t_logprobs, t_idxs = torch.topk(logprobs, n_beams)
      t_logprobs, t_idxs = t_logprobs[0].tolist(), t_idxs[0].tolist()
      for lp, idx in zip(t_logprobs,t_idxs):
        nlp = logp+lp
        nys = torch.cat([ys, torch.tensor([[idx]])], dim=0)
        if idx==EOS_IDX or idx==13: top_sentences.add((nlp,nys))
        else: h.add((nlp, nys))
    beams = h.getTop()
  
  preds = [(lp,tokenizer.decode(idxs.T[0].tolist()).strip()) for lp, idxs in top_sentences.getTop()]
  return preds


MODEL_LOAD_NAME = 'model_eos'
model = RefTransformer(tokenizer.get_vocab_size())
model.load_state_dict(torch.load('./models/' + MODEL_LOAD_NAME + '.pt'))

_, xys_val = load_datasets()

for x,y in xys_val[:1]:
  preds = beam_search_decode(model, x)
  print(x)
  print(y)
  for lp, pred in preds:
    print(round(lp,3), pred)
  print()

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
  pred = greedy_decode(model, x)
  preds.append([x,y,pred])

writelines("./out/orig.txt", [x for x,y,p in preds])
writelines("./out/corr.txt", [y for x,y,p in preds])
writelines("./out/pred.txt", [p for x,y,p in preds])
