import torch
from dataloader import *
from model import RefTransformer
from mytokenizer import BOS_IDX, EOS_IDX, to_token_idxs, tokenizer


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

print(tokenizer.encode('a.').ids)
_, xys_val = load_datasets()
random.shuffle(xys_val)
orig, corr = xys_val[0]
xys_val = to_token_idxs(xys_val)
src, tgt = xys_val[0]

idxs = greedy_decode(model, torch.tensor([src]).T, 100)
pred = tokenizer.decode(idxs.T[0].tolist()).strip()

print(orig)
print(corr)
print(pred)

def correct_sentence(sentence):
  src = tokenizer.encode(sentence).ids
  idxs = greedy_decode(model, torch.tensor([src]).T, 100)
  pred = tokenizer.decode(idxs.T[0].tolist()).strip()
  return pred


sentence = 'It is important to consider the long term effects.'
print(correct_sentence(sentence))