import torch
from dataloader import get_orig_and_corr
from model import EncoderClassifier
from mytokenizer import PAD_IDX, tokenizer
from utils import count_params, noise, to_padded_tensor, unzip

VOCAB_S = tokenizer.get_vocab_size()

orig, corr = get_orig_and_corr()
sentences = orig + corr

xys = [noise(x.ids, VOCAB_S) for x in tokenizer.encode_batch(sentences)]
xs_test, ys_test = unzip(xys[:500])
xs_train, ys_train = unzip(xys[500:628])
print(len(xs_train), len(xs_train[0]))

model = EncoderClassifier(tokenizer.get_vocab_size(), 256, 3)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
print(count_params(model))
BATCH_SIZE = 128
EPOCHS = 100


for epoch in range(EPOCHS):
  i = 0
  tot_loss = 0
  print('epoch', epoch+1)
  while i + BATCH_SIZE <= len(xs_train):
    xs = to_padded_tensor(xs_train[i:i+BATCH_SIZE], PAD_IDX)
    ys = to_padded_tensor(ys_train[i:i+BATCH_SIZE], PAD_IDX)
    ys_pred = model(xs)

    optimizer.zero_grad()
    loss = loss_fn(ys_pred.view(-1, ys_pred.shape[-1]), ys.view(-1))
    optimizer.step()
    tot_loss += loss.item()
    i+=BATCH_SIZE
    print(i, len(xs_train), "%0.4f" % loss.item())

