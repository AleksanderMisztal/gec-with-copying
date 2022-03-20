import torch
from copygec.models.transformer_custom import CopyEncoderDecoder
from copygec.mytokenizer import BOS_IDX, EOS_IDX, dec

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
