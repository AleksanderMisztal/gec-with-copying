import torch
from model import RefTransformer
from dataloader import DataLoader, get_orig_corr_pairs
from mytokenizer import PAD_IDX, to_token_idxs, tokenizer
from utils import create_mask, get_tf_predictions

BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = RefTransformer(tokenizer.get_vocab_size())
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.3, threshold=.1, verbose=True)

xys = to_token_idxs(get_orig_corr_pairs(lim=80))

xys_test = xys[:BATCH_SIZE]
xys_train = xys[BATCH_SIZE:]

test_dataloader = DataLoader(xys_test, BATCH_SIZE, PAD_IDX)
train_dataloader = DataLoader(xys_train, BATCH_SIZE, PAD_IDX)

def train_epoch(model, loss_fn, train_dataloader):
  model.train()
  losses = 0
  steps = 0

  for src, tgt in train_dataloader:
    steps += 1
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX)

    logits = model(src, tgt_input)#, tgt_mask, src_padding_mask, tgt_padding_mask)
    optimizer.zero_grad()

    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()

    optimizer.step()
    scheduler.step(loss.item())
    losses += loss.item()
    print(losses/steps)

  return losses / steps

def evaluate(model, loss_fn, test_dataloader):
  model.eval()
  losses = 0
  steps = 0

  for src, tgt in test_dataloader:
    steps += 1
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX)

    logits = model(src, tgt_input)

    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

    losses += loss.item()

  return losses / steps


transformer.load_state_dict(torch.load('./models/model.pt'))
min_loss = evaluate(transformer, loss_fn, test_dataloader)
print('min loss is:', min_loss)
for i in range(50):
  av_loss = train_epoch(transformer, loss_fn, train_dataloader)
  eval_loss = evaluate(transformer, loss_fn, test_dataloader)
  print(i, av_loss, eval_loss)

  sps = xys_test[:4]
  sps_pred = get_tf_predictions(transformer, sps, PAD_IDX)
  sps_pred = tokenizer.decode_batch(sps_pred)
  for i in range(1):
    print(i)
    sp = tokenizer.decode_batch(sps[i])
    print(sp[0])
    print(sp[1])
    print(sps_pred[i])
  if eval_loss < min_loss:
    torch.save(transformer.state_dict(), './models/model.pt')
    min_loss = eval_loss
    print('saved!!')


# TODO pretrain the encoder on word prediction
# TODO group into batches based on length
# TODO train on google colab
# TODO split into encoder - decoder
# TODO use the masks properly

