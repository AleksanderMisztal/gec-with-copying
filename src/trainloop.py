import os
import torch
from pathlib import Path
from argparse import ArgumentParser

from copygec.model import RefTransformer
from copygec.dataloader import DataLoader, load_datasets
from copygec.mytokenizer import PAD_IDX, to_token_idxs, tokenizer
from copygec.keyinterrupt import prevent_interrupts, was_interrupted

LOCAL_DEV = os.environ['GEC_ENV'] != "hpc"

parser = ArgumentParser()
parser.add_argument("--ln", "--loadname", dest="loadname",
                    help="Load model with this name")
parser.add_argument("--sn", "--savename", dest="savename",
                    help="Save with this name", required=True)
args = parser.parse_args()

IS_MODEL_LOADED = args.loadname is not None
if IS_MODEL_LOADED: MODEL_LOAD_PATH = './models/transformer/' + args.loadname + '.pt'
MODEL_SAVE_PATH = './models/transformer/' + args.savename + '.pt'

print(IS_MODEL_LOADED, MODEL_LOAD_PATH, MODEL_SAVE_PATH)
if not Path(MODEL_LOAD_PATH).exists(): 
  print('Attmepting to load a model that does not exist!')
  exit()

xys_train, xys_val = load_datasets('./data/')
xys_train = to_token_idxs(xys_train)
xys_val = to_token_idxs(xys_val)
print("Train / val set sizes:", len(xys_train), len(xys_val))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = .001
BATCH_SIZE = 128
train_dataloader = DataLoader(xys_train, BATCH_SIZE, PAD_IDX)
test_dataloader = DataLoader(xys_val, BATCH_SIZE, PAD_IDX)

transformer = RefTransformer(tokenizer.get_vocab_size(), device=DEVICE)
if IS_MODEL_LOADED: transformer.load_state_dict(torch.load(MODEL_LOAD_PATH))
print("Device being used:", DEVICE)
print("Is model cuda?", next(transformer.parameters()).is_cuda)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.3, threshold=.01, verbose=True, patience=10)

def train_epoch(model, loss_fn, train_dataloader, verbose=False):
  model.train()
  losses = 0
  steps = 0

  for src, tgt in train_dataloader:
    steps += 1
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    logits = model(src, tgt_input)
    optimizer.zero_grad()

    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()

    optimizer.step()
    losses += loss.item()
    if verbose: print(f'Step {steps*BATCH_SIZE} / {len(train_dataloader)}. Train loss: {round(losses/steps,3)}', end='\r')
    if was_interrupted(): return losses/steps

  scheduler.step(losses/steps)
  return losses/steps

def evaluate(model, loss_fn, test_dataloader, lim=-1):
  model.eval()
  losses = 0
  steps = 0

  for src, tgt in test_dataloader:
    steps += 1
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    logits = model(src, tgt_input)

    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

    losses += loss.item()
    if lim > 0 and steps >= lim: break

  return losses / steps

Path('./models/transformer').mkdir(parents=True, exist_ok=True)
def save_model():
  torch.save(transformer.state_dict(), MODEL_SAVE_PATH)
  print('Saved!')

# Evaluate on one batch
min_loss = evaluate(transformer, loss_fn, test_dataloader, lim=1)
print('Initial validation loss:', round(min_loss,3))

EPOCHS = 5
prevent_interrupts()
for i in range(1, EPOCHS+1):
  train_loss = train_epoch(transformer, loss_fn, train_dataloader, True)
  if was_interrupted():
    save_model()
    exit()
  eval_loss = evaluate(transformer, loss_fn, test_dataloader, lim=3)
  print(f'Epoch {i} done. t: {round(train_loss,3)}, v: {round(eval_loss,3)}.',end=' ')
  save_model()


# TODO fix all to be on the same device
# TODO Run the model on HPC
# TODO Faster decoding

# TODO Benchmark the HPC
# TODO Will model eval give better results? (dropout)
# TODO Evaluate sythetic data pretraining
# TODO Evaluate beam search
# TODO Speed up beam search (batching?)
# TODO Test inference speed varying n_layers, batching, decoding, etc.
# TODO Errant on write and improve data
# TODO Start documenting
# TODO Increase model size
# TODO Try w/wout beam search normalizing