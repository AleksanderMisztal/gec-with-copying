import torch
import os
import json
from pathlib import Path

from copygec.dataloader import DataLoader, sentences_to_padded_tensor
from copygec.decoding import greedy_decode
from copygec.training import train_model as _train_model
from copygec.utils import noise, writelines, read_json
from copygec.models.optimizer import get_std_opt
from copygec.mytokenizer import PAD_IDX, enc


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128


def get_save_path(model_name):
  return './models/transformer/' + model_name + '.pt'

def load_model(model, model_name):
  save_path = get_save_path(model_name)
  model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

def train_model(model, xys_train, xys_dev, epochs, model_name, add_noise=False):
  save_path = get_save_path(model_name)
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
  optimizer = get_std_opt(model, model.d_model)
  preprocess = noise if add_noise else None

  train_dataloader = DataLoader(xys_train, BATCH_SIZE, DEVICE, preprocess)
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
  corrections = [e for e in ocps if e['corr'] == e['pred'] and e['pred'] != e['orig']]
  with open(dir+'/results.json', 'w', encoding='utf-8') as f:
    json.dump(ocps, f, ensure_ascii=False, indent=2)
  with open(dir+'/mistakes.json', 'w', encoding='utf-8') as f:
    json.dump(mistakes, f, ensure_ascii=False, indent=2)
  with open(dir+'/corrections.json', 'w', encoding='utf-8') as f:
    json.dump(corrections, f, ensure_ascii=False, indent=2)

def run_errant(model_name):
  print('Running errant...', flush=True)
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

from copygec.mytokenizer import ids_to_tokens

def visualise_distribution(a, copy_probs, gen_probs, true_ids):
  copy = torch.topk(copy_probs, 5).indices
  gen = torch.topk(gen_probs, 5).indices
  gen_tokens = [ids_to_tokens(ids) for ids in gen]
  copy_tokens = [ids_to_tokens(ids) for ids in copy]
  true_tokens = ids_to_tokens(true_ids)
  a = a.detach().numpy()

  copied = False
  for tt, ai, p_copy, p_gen in zip(true_tokens, a, copy_tokens, gen_tokens):
    air = round(ai,2)
    if ai > 0: copied = True
    print(tt, air, p_copy, 1-air, p_gen)
  if copied: 
    print("COPYYYYYYYYYYYYYY")
    exit()

def visualise_copying(model, xys):
  for x, y in xys:
    print(x)
    print(y)
    print(enc(y))
    src = sentences_to_padded_tensor([x])
    tgt = sentences_to_padded_tensor([y])
    tgt_in =  tgt[:-1, :]
    tgt_out =  tgt[1:, :]
    out = model(src, tgt_in)
    data = model.generator.copy_data
    a = data['a'][:,0,0]
    copy = data['copy'][:,0,:]
    gen = data['gen'][:,0,:]
    print(greedy_decode(model, src)[0])
    visualise_distribution(a, copy, gen, tgt_out)
  