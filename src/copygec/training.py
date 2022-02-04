from copygec.dataloader import DataLoader


def train_epoch(model, loss_fn, dataloader: DataLoader, optimizer, device, verbose=False, history=False):
  model.train()
  losses = 0
  steps = 0
  loss_history = []

  for src, tgt in dataloader:
    steps += 1
    src = src.to(device)
    tgt = tgt.to(device)
    tgt_input = tgt[:-1, :]
    logits = model(src, tgt_input)
    optimizer.zero_grad()
    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()
    optimizer.step()
    losses += loss.item()
    loss_history.append(loss.item())

    if verbose: print(f'Step {min(len(dataloader), steps*dataloader.batch_size)} / {len(dataloader)}. Train loss: {round(losses/steps,3)}.                          ',end='\r')

  if history: return losses/steps, loss_history
  return losses/steps

def evaluate(model, loss_fn, dataloader, device):
  model.eval()
  losses = 0
  steps = 0

  for src, tgt in dataloader:
    steps += 1
    src = src.to(device)
    tgt = tgt.to(device)
    tgt_input = tgt[:-1, :]
    logits = model(src, tgt_input)
    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    losses += loss.item()

  return losses / steps
