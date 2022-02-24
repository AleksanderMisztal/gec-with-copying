import torch
import os
import json
from pathlib import Path

from copygec.dataloader import DataLoader, sentences_to_padded_tensor
from copygec.decoding import greedy_decode
from copygec.training import train_model as _train_model
from copygec.utils import writelines, read_json
from copygec.models.optimizer import get_std_opt
from copygec.mytokenizer import PAD_IDX


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128


def get_save_path(model_name):
  return './models/transformer/' + model_name + '.pt'

def load_model(model, model_name):
  save_path = get_save_path(model_name)
  model.load_state_dict(torch.load(save_path))

def train_model(model, xys_train, xys_dev, epochs, model_name):
  save_path = get_save_path(model_name)
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  optimizer = get_std_opt(model, model.d_model)

  train_dataloader = DataLoader(xys_train, BATCH_SIZE, DEVICE)
  dev_dataloader = DataLoader(xys_dev, BATCH_SIZE, DEVICE)

  _train_model(model, loss_fn, train_dataloader, dev_dataloader, optimizer, epochs, save_path)

def get_predictions(model, xs):
  s, bs = len(xs), BATCH_SIZE
  batches = [xs[i*bs:(i+1)*bs] for i in range(s//bs)]
  if s % bs != 0: batches.append(xs[s//bs*bs:])
  pred = []
  for batch in batches:
    src = sentences_to_padded_tensor(batch).to(DEVICE)
    pred += greedy_decode(model, src)
  return pred

def save_results(orig, corr, pred, model_name):
  dir = './out/' + model_name
  Path(dir).mkdir(parents=True, exist_ok=True)

  ocps = [{'corr': c, 'orig': o, 'pred': p} for (c, o, p) in zip (corr, orig, pred)]
  mistakes = [entry for entry in ocps if entry['corr'] != entry['pred']]
  with open(dir+'/results.json', 'w', encoding='utf-8') as f:
    json.dump(ocps, f, ensure_ascii=False, indent=2)
  with open(dir+'/mistakes.json', 'w', encoding='utf-8') as f:
    json.dump(mistakes, f, ensure_ascii=False, indent=2)

def run_errant(model_name):
  dir = './out/' + model_name
  ocps = read_json(dir+'/results.json')
  
  writelines(dir+"/orig.txt", [r['orig'] for r in ocps])
  writelines(dir+"/corr.txt", [r['corr'] for r in ocps])
  writelines(dir+"/pred.txt", [r['pred'] for r in ocps])

  get_ref = f"errant_parallel -orig {dir}/orig.txt -cor {dir}/corr.txt -out {dir}/ref.m2"
  get_hyp = f"errant_parallel -orig {dir}/orig.txt -cor {dir}/pred.txt -out {dir}/hyp.m2"
  compare = f"errant_compare -hyp {dir}/hyp.m2 -ref {dir}/ref.m2"

  os.system(get_ref)
  os.system(get_hyp)
  os.system(compare)

  os.system('rm ./out/' +model_name+ '/{orig,corr,pred}.txt')
  os.system('rm ./out/' +model_name+ '/{hyp,ref}.m2')