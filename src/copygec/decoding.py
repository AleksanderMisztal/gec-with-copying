import torch
from copygec.mytokenizer import BOS_IDX, EOS_IDX, dec, sentence_to_tensor
from copygec.utils import MaxHeap

logSoftmax = torch.nn.functional.log_softmax

def greedy_decode(model, sentence, max_len=100):
  src = sentence_to_tensor(sentence).unsqueeze(1)
  memory = model.encode(src)
  ys = torch.tensor([[BOS_IDX]])
  logprob = 0
  for _ in range(max_len):
    out = model.decode(ys, memory)[0][0]
    probs = logSoftmax(out, dim=0)
    assert len(probs.shape) == 1
    n_logprob, next_word = torch.max(probs, dim=0)
    next_word = next_word.item()
    logprob += n_logprob.item()
    ys = torch.cat([ys, torch.tensor([[next_word]])], dim=0)
    if next_word == EOS_IDX: break
  pred = dec(ys[:,0].tolist())
  return logprob, pred

def beam_search_decode(model, sentence, n_beams=12, n_results=12, max_len=10):
  src = sentence_to_tensor(sentence)
  memory = model.encode(src)
  top_sentences = MaxHeap(n_results)
  beams = [(0,torch.tensor([[BOS_IDX]])) for _ in range(n_beams)]
  for _ in range(max_len):
    print(f'step {_}')
    h = MaxHeap(n_beams)
    for logp,ys in beams:
      print('logp, ys:', logp, ys.T)
      out = model.decode(ys, memory)[0][0]
      logprobs = logSoftmax(out, dim=0)
      # print('logprobs:',logprobs.shape)
      t_logprobs, t_idxs = torch.topk(logprobs, n_beams)
      # print(t_logprobs.shape)
      t_logprobs, t_idxs = t_logprobs.tolist(), t_idxs.tolist()
      for lp, idx in zip(t_logprobs,t_idxs):
        nlp = logp+lp
        nys = torch.cat([ys, torch.tensor([[idx]])], dim=0)
        if idx==EOS_IDX: top_sentences.add((nlp,nys))
        else: h.add((nlp, nys))
    beams = h.getTop()
  preds = [(lp,dec(idxs.T[0].tolist())) for lp, idxs in top_sentences.getTop()]
  return preds
