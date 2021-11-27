import torch
from mytokenizer import BOS_IDX, EOS_IDX, tokenizer
from utils import MaxHeap

logSoftmax = torch.nn.LogSoftmax(dim=1)

def get_tf_predictions(model, src: 'list[int]', tgt: 'list[int]'):
  src = torch.tensor([src]).T
  tgt = torch.tensor([tgt]).T
  out = model(src, tgt[:-1, :])
  words = torch.argmax(out, dim=2)
  idxs = words.T.tolist()[0]
  pred = tokenizer.decode(idxs)
  return pred

def greedy_decode(model, sentence, max_len=100):
  src = tokenizer.encode(sentence).ids
  src = torch.tensor([src]).T
  memory = model.encode(src)
  ys = torch.tensor([[BOS_IDX]])
  prob = 0
  for i in range(max_len):
    out = model.decode(ys, memory)
    out = out.transpose(0, 1)
    probs = logSoftmax(model.generator(out[:, -1]))
    n_logprob, n_word = torch.max(probs, dim=1)
    n_word = n_word.item()
    prob += n_logprob.item()
    ys = torch.cat([ys, torch.tensor([[n_word]])], dim=0)
    if n_word == EOS_IDX or n_word==13: break
  
  pred = tokenizer.decode(ys.T[0].tolist()).strip()
  return prob, pred

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
      logprobs = logSoftmax(model.generator(out[:, -1]))
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
