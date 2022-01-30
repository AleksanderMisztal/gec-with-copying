import torch
from random import SystemRandom
random = SystemRandom()
from copygec.dataloader import DataLoader
from copygec.models.transformer_ref import Transformer
from copygec.training import evaluate, train_epoch

vocab_s = 100
BOS_IDX=vocab_s
EOS_IDX=vocab_s+1
PAD_IDX=vocab_s+2

def rw():
  return random.randint(0, vocab_s-1)

xys = [([5,5,5,5,5],[BOS_IDX]+[rw() for i in range(2)]*2+[EOS_IDX]) for _ in range(20480)]
xys_val = xys[:1024]
xys_train = xys[1024:]

for s, t in xys_val[:5]:
  print(s)
  print(t)
  print()

train_dataloader = DataLoader(xys_train, 128, PAD_IDX)
val_dataloader = DataLoader(xys_val, 128, PAD_IDX)
print("Data loaded! Train / val set sizes:",len(xys_train), len(xys_val))

transformer = Transformer(vocab_s+3, PAD_IDX, num_layers=2)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

for i in range(20):
  train_loss = train_epoch(transformer, loss_fn, train_dataloader, optimizer, torch.device('cpu'), True)
  eval_loss = evaluate(transformer, loss_fn, val_dataloader, torch.device('cpu'))
  print(f'Epoch {i} done. T: {round(train_loss,3)}, v: {round(eval_loss,3)}.')

# for src, tgt in val_dataloader:
#     tgt_input = tgt[:-1, :]
#     logits = transformer(src, tgt_input)
#     tgt_out = tgt[1:, :]
#     loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
#     print('loss:', loss)
#     print('logits:', logits)
#     print('tgt_out:', tgt_out)

# Somehow this still gets down to 3.5 = ln((100+0+0)/3). =
# 3.45 = 3/4 * ln(100)
# Should stay at 4.605 = ln(100) right?
# With 5 words htis gives 3.855 ~ ln(47), wtf?
# ln(100/5) = 3
# ln(100/2) = 3.9