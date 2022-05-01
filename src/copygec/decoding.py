import math
import torch
from copygec.models.transformer_custom import CopyEncoderDecoder
from copygec.mytokenizer import BOS_IDX, EOS_IDX, dec
from copygec.utils import MaxHeap

logSoftmax = torch.nn.functional.log_softmax

def greedy_decode(model, src, max_len=100):
  model.eval()
  is_copy = type(model) is CopyEncoderDecoder
  bs = src.shape[1]
  memory = model.encode(src)
  ys = torch.tensor([[BOS_IDX]*bs]).to(model.device)
  done = torch.zeros(bs).to(model.device)
  for i in range(max_len):
    out = model.decode(ys, memory, src) if is_copy else model.decode(ys, memory)
    out = out[i:i+1,:,:]
    log_probs = logSoftmax(out, dim=2)
    _, next_words = torch.max(log_probs, dim=2)
    ys = torch.cat([ys, next_words], dim=0)
    done += (next_words[0] == EOS_IDX)
    if torch.all(done) == bs: break
  idss = ys.T.tolist()
  
  for ids in idss: ids.append(EOS_IDX)
  idss = [ids[:ids.index(EOS_IDX)] for ids in idss]
  pred = [dec(ids) for ids in idss]
  
  return pred


def beam_search_decode(model, src, n_beams=12, n_results=12, max_len=30):
  memory = model.encode(src)
  top_sentences = MaxHeap(n_results)
  beams = [(0,torch.tensor([[BOS_IDX]]))]
  for curr_len in range(max_len):
    print(f'step {curr_len}')
    h = MaxHeap(n_beams)
    for logp,ys in beams:
      # print('logp, ys:', logp, ys.T)
      out = model.decode(ys, memory, src)[curr_len, 0, :]
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
    input('continue?')
    print('top beams')
    print(math.exp(beams[0][0]), dec(beams[0][1].T[0].tolist()))
    print(math.exp(beams[1][0]), dec(beams[1][1].T[0].tolist()))
    print(math.exp(beams[2][0]), dec(beams[2][1].T[0].tolist()))
  preds = [(lp,dec(idxs.T[0].tolist())) for lp, idxs in top_sentences.getTop()]
  return preds