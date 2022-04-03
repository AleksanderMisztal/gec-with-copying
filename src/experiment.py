import torch

from copygec.dataloader import DataLoader, load_datasets
from copygec.mytokenizer import PAD_IDX, VOCAB_S
import torch.nn.functional as F


def evaluate(loss_fn, dataloader):
  losses = 0
  steps = 0

  for src, tgt in dataloader:
    steps += 1
    tgt_out = F.one_hot(tgt[1:, :], VOCAB_S)
    print(src.shape, tgt_out.shape)
    loss = loss_fn(src.reshape(-1, src.shape[-1]), tgt_out.reshape(-1))
    losses += loss.item()

  return losses / steps

xys = load_datasets()
print(len(xys['train']),len(xys['dev']),len(xys['test']))
dataloader = DataLoader(xys['dev'], 128, torch.device('cpu'))
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

print(evaluate(loss_fn, dataloader))