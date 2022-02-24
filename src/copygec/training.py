import json
import torch
import os
from pathlib import Path
from copygec.dataloader import DataLoader
from copygec.decoding import greedy_decode
from copygec.mytokenizer import PAD_IDX, sentence_to_tokens, to_token_idxs
from copygec.utils import to_padded_tensor, unzip, writelines


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

def train_model(model, xys_train, xys_dev, optimizer, batch_size, epochs, device, model_name):
  print('Starting training...')

  MODEL_SAVE_PATH = './models/transformer/' + model_name + '.pt'
  Path('./models/transformer').mkdir(parents=True, exist_ok=True)

  train_dataloader = DataLoader(to_token_idxs(xys_train), batch_size, PAD_IDX)
  dev_dataloader = DataLoader(to_token_idxs(xys_dev), batch_size, PAD_IDX)
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

  min_dev_loss = 100_000
  for i in range(1, epochs+1):
    train_loss = train_epoch(model, loss_fn, train_dataloader, optimizer, device, True)
    dev_loss = evaluate(model, loss_fn, dev_dataloader, device)
    print(f'Epoch {i} done. Train loss: {round(train_loss,3)}, dev loss: {round(dev_loss,3)}.',end=' ')
    if dev_loss < min_dev_loss:
      min_dev_loss = dev_loss
      torch.save(model.state_dict(), MODEL_SAVE_PATH)
      print('Saved!')
    else: print()

def write_for_evaluation(model, xys, bs, device, name):
  print(f'Writing predictions for {len(xys)} sentences...')
  xys.sort(key=lambda xy: len(xy[0]))
  s = len(xys)
  batches = [xys[i*bs:(i+1)*bs] for i in range(s//bs)]
  if s % bs != 0: batches.append(xys[s//bs*bs:])
  preds = []
  for batch in batches:
    xs, ys = unzip(batch)
    src = to_padded_tensor([sentence_to_tokens(x) for x in xs], PAD_IDX).T
    src = src.to(device)
    batch_preds = greedy_decode(model, src, device)
    preds+=batch_preds
  
  write_path = './out/'+name
  Path(write_path).mkdir(parents=True, exist_ok=True)

  xs = [x for x,y in xys]
  ys = [y for x,y in xys]
  ocps = [{'corr': corr, 'orig': orig, 'pred': pred} for (corr, orig, pred) in zip (ys, xs, preds)]

  writelines(write_path+"/orig.txt", xs)
  writelines(write_path+"/corr.txt", ys)
  writelines(write_path+"/pred.txt", preds)
  with open(write_path+'/results.json', 'w', encoding='utf-8') as f:
    json.dump(ocps, f, ensure_ascii=False, indent=2)

  print(f'Done!')

def run_errant(in_dir):
  print('Starting to evaluate with errant...')

  get_ref = f"errant_parallel -orig {in_dir}/orig.txt -cor {in_dir}/corr.txt -out {in_dir}/ref.m2"
  get_hyp = f"errant_parallel -orig {in_dir}/orig.txt -cor {in_dir}/pred.txt -out {in_dir}/hyp.m2"
  compare = f"errant_compare -hyp {in_dir}/hyp.m2 -ref {in_dir}/ref.m2"

  os.system(get_ref)
  os.system(get_hyp)
  os.system(compare)
