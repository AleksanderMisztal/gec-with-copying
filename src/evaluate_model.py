from pathlib import Path
import random
import torch
from argparse import ArgumentParser

from copygec.dataloader import load_datasets
from copygec.decoding import greedy_decode
from copygec.models.transformer_ref import Transformer
from copygec.mytokenizer import PAD_IDX, sentence_to_tokens, tokenizer
from copygec.utils import to_padded_tensor, unzip, writelines

BATCH_SIZE=128

def write_for_evaluation(model, xys):
  print(f'Writing predictions for {len(xys)} sentences...')
  xys.sort(key=lambda xy: len(xy[0]))
  s, bs = len(xys), BATCH_SIZE
  batches = [xys[i*bs:(i+1)*bs] for i in range(s//bs)]
  if s % bs != 0: batches.append(xys[s//bs*bs:])
  preds = []
  for batch in batches:
    xs, ys = unzip(batch)
    src = to_padded_tensor([sentence_to_tokens(x) for x in xs], PAD_IDX).T
    batch_preds = greedy_decode(model, src)
    preds+=batch_preds

  writelines("./out/orig.txt", [x for x,y in xys])
  writelines("./out/corr.txt", [y for x,y in xys])
  writelines("./out/pred.txt", preds)

  print(f'Done!')

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--ln", "--loadname", dest="loadname",
                    help="Load model with this name")
  parser.add_argument("-n", "--nsentences", dest="nsentences",
                      help="Number of sentences to be processed", type=int)
  parser.add_argument("--layers", dest="layers", type=int,
                    help="Create this many layers in the transformer", required=True)
  args = parser.parse_args()

  loadpath = './models/transformer/' + args.loadname + '.pt'
  if not Path(loadpath).exists():
    print('Attmepting to load a model that does not exist!')
    exit()
  
  _, xys_val = load_datasets('./data/')
  if args.nsentences is not None and args.nsentences > len(xys_val):
    print(f"Not enough sentences in val set")
    exit()

  model = Transformer(tokenizer.get_vocab_size(), PAD_IDX, num_layers=args.layers)
  model.load_state_dict(torch.load(loadpath))

  if args.nsentences is None: xys = xys_val
  else: xys = random.sample(xys_val, args.nsentences)


  write_for_evaluation(model, xys)