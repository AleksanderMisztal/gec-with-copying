import random
import torch
from copygec.decoding import get_tf_predictions
from copygec.model import RefTransformer
from copygec.dataloader import DataLoader, load_datasets
from copygec.mytokenizer import PAD_IDX, to_token_idxs, tokenizer
from copygec.keyinterrupt import prevent_interrupts, was_interrupted, interrupt_handled
from pathlib import Path

xys_train, xys_val = load_datasets('./data/')
xys_train = to_token_idxs(xys_train)
xys_val = to_token_idxs(xys_val)
print(len(xys_train), len(xys_val))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = .001
BATCH_SIZE = 128
train_dataloader = DataLoader(xys_train, BATCH_SIZE, PAD_IDX)
test_dataloader = DataLoader(xys_val, BATCH_SIZE, PAD_IDX)

transformer = RefTransformer(tokenizer.get_vocab_size())
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.3, threshold=.01, verbose=True, patience=10)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: min((e+1)**(-1/2), (e+1)*(40**(-3/2)))/160)

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

def evaluate(model, loss_fn, test_dataloader):
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

  return losses / steps

def save_model():
  torch.save(transformer.state_dict(), './models/' + MODEL_SAVE_NAME + '.pt')
  print('Saved!')

Path('./models/transformer').mkdir(parents=True, exist_ok=True)
MODEL_LOAD_NAME = 'transformer/model'
MODEL_SAVE_NAME = 'transformer/model_wi'
IS_MODEL_LOADED = False

def confirm(message):
  cont = input(message + ' (y/n) ').strip()
  if cont != 'y': exit()

confirm(f'Are you sure you want to save the model as {MODEL_SAVE_NAME}?')

if IS_MODEL_LOADED and MODEL_LOAD_NAME != MODEL_SAVE_NAME:
  confirm('Are you sure you want to load from a different file?')

if IS_MODEL_LOADED: transformer.load_state_dict(torch.load('./models/' + MODEL_LOAD_NAME + '.pt'))
else:
  confirm('Are you sure you want to initialize a new model?')  
min_loss = evaluate(transformer, loss_fn, test_dataloader)
print('Initial validation loss:', round(min_loss,3))

def visualise_model(n: int):
  sps = random.sample(xys_val, n)
  preds = [(*tokenizer.decode_batch(sp), get_tf_predictions(transformer, *sp)) for sp in sps]
  preds = [[s.strip() for s in xyp] for xyp in preds]
  for x, y, y_pred in preds:
    print(x)
    print(y)
    print(y_pred)

def handle_training_interrupt():
  action = input('c -> continue\nh -> / 10 learning rate\nx -> 10x learning rate\nq -> quit\ns -> save\nv -> visualize\n').strip()
  if action == 'c': pass
  elif action == 'h':
    for g in optimizer.param_groups:
      g['lr'] /= 10
    print('lr halved')
    print('new lr =', optimizer.param_groups[0]['lr'])
  elif action == 'x':
    for g in optimizer.param_groups:
      g['lr'] *= 10
    print('lr 10xed')
    print('new lr =', optimizer.param_groups[0]['lr'])
  elif action == 'q':
    save_model()
    exit()
  elif action == 's':
    save_model()
  elif action == 'v':
    visualise_model(4)
  interrupt_handled()

EPOCHS = 1000
prevent_interrupts()
for i in range(1, EPOCHS+1):
  train_loss = train_epoch(transformer, loss_fn, train_dataloader, True)
  if was_interrupted(): 
    handle_training_interrupt()
    continue
  eval_loss = evaluate(transformer, loss_fn, test_dataloader)
  print(f'Epoch {i} done. t: {round(train_loss,3)}, v: {round(eval_loss,3)}.',end=' ')
  save_model()

# TODO Benchmark the HPC GPUs
# TODO Will model eval give better results? (dropout)
# TODO Evaluate sythetic data pretraining
# TODO Evaluate beam search
# TODO Speed up beam search (batching?)
# TODO Test inference speed varying n_layers, batching, decoding, etc.
# TODO Errant on write and improve data
# TODO Start documenting
# TODO Increase model size
# TODO Try w/wout beam search normalizing