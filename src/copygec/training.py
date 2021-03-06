import torch

def train_epoch(model: torch.nn.Module, loss_fn, dataloader, optimizer):
  model.train()
  losses = 0
  steps = 0
  for src, tgt in dataloader:
    steps += 1
    tgt_input = tgt[:-1, :]
    logits = model(src, tgt_input)
    optimizer.zero_grad()
    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()
    optimizer.step()
    losses += loss.detach().item()
    del loss
    del logits
  optimizer.zero_grad()
  average_loss = losses / steps
  assert type(average_loss) is float
  return average_loss


def evaluate(model, loss_fn, dataloader):
  model.eval()
  losses = 0
  steps = 0
  for src, tgt in dataloader:
    steps += 1
    tgt_input = tgt[:-1, :]
    logits = model(src, tgt_input)
    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    losses += loss.detach().item()
    
  average_loss = losses / steps
  assert type(average_loss) is float
  return average_loss


def train_model(model, loss_fn, train_dataloader, dev_dataloader, optimizer, epochs, save_path):
  min_dev_loss = 100_000
  for i in range(1, epochs+1):
    train_loss = train_epoch(model, loss_fn, train_dataloader, optimizer)
    dev_loss = evaluate(model, loss_fn, dev_dataloader)
    print(f'Epoch {i} done. Train loss: {round(train_loss,3)}, dev loss: {round(dev_loss,3)}.',end='')
    if dev_loss < min_dev_loss:
      min_dev_loss = dev_loss
      torch.save(model.state_dict(), save_path)
      print(' Saved!', flush=True)
    else: print('', flush=True)
