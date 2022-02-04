import torch
from random import SystemRandom
random = SystemRandom()
from copygec.dataloader import DataLoader
#from copygec.models.transformer_ref import Transformer
from copygec.models.transformer_custom import make_model as Transformer
from copygec.training import evaluate, train_epoch

vocab_s = 100
BOS_IDX=vocab_s
EOS_IDX=vocab_s+1
PAD_IDX=vocab_s+2

def rw(): return random.randint(0, vocab_s-1)

xys = [([BOS_IDX,w1,w2,EOS_IDX],[BOS_IDX,w3,w4,w5,w3,w4,w5,EOS_IDX]) for _ in range(20_000) for w1, w2, w3, w4, w5 in [[rw() for _ in range(5)]]]
xys_val = xys[:1024]
xys_train = xys[1024:]

train_dataloader = DataLoader(xys_train, 128, PAD_IDX)
val_dataloader = DataLoader(xys_val, 128, PAD_IDX)
print("Data loaded! Train / val set sizes:",len(xys_train), len(xys_val))

transformer = Transformer(vocab_s+3, PAD_IDX, num_layers=2, d_model=256)#, copy=True)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)

for x, y in train_dataloader:
  print(x[:,0])
  print(y[:,0])
  print(transformer(x, y[:-1]).shape)
  break

losses = []
for i in range(2):
  train_loss, history = train_epoch(transformer, loss_fn, train_dataloader, optimizer, torch.device('cpu'), verbose=True, history=True)
  losses += history
  eval_loss = evaluate(transformer, loss_fn, val_dataloader, torch.device('cpu'))
  print(f'Epoch {i} done. T: {round(train_loss,3)}, v: {round(eval_loss,3)}.                    ')

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()