import random
import torch
from model import RefTransformer
from dataloader import DataLoader, load_datasets
from mytokenizer import PAD_IDX, to_token_idxs, tokenizer
from utils import get_tf_predictions
from keyinterrupt import prevent_interrupts, was_interrupted, interrupt_handled

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
TRAIN_S = 2048
VAL_S = 256
LEARNING_RATE = .01

xys_train, xys_val = load_datasets()
xys_train = to_token_idxs(xys_train[:TRAIN_S])
xys_val = to_token_idxs(xys_val[:VAL_S])

test_dataloader = DataLoader(xys_val, BATCH_SIZE, PAD_IDX)
train_dataloader = DataLoader(xys_train, BATCH_SIZE, PAD_IDX)

transformer = RefTransformer(tokenizer.get_vocab_size())
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.3, threshold=.01, verbose=True)

def train_epoch(model, loss_fn, train_dataloader):
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
    #scheduler.step(loss.item())
    losses += loss.item()
    print(f'Step {steps*BATCH_SIZE} / {len(train_dataloader)}. Train loss: {round(losses/steps,3)}', end='\r')

    if was_interrupted(): return losses/steps

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

MODEL_LOAD_NAME = 'modelsm'
MODEL_SAVE_NAME = 'model_xxs'
IS_MODEL_LOADED = True

cont = input(f'Are you sure you want to save the model as {MODEL_SAVE_NAME}? (y/n) ').strip()
if cont != 'y': exit()

if IS_MODEL_LOADED:
  transformer.load_state_dict(torch.load('./models/' + MODEL_LOAD_NAME + '.pt'))
else:
  cont = input('Are you sure you want to initialize a new model? (y/n) ').strip()
  if cont != 'y': exit()
min_loss = evaluate(transformer, loss_fn, test_dataloader)
print('Initial validation loss:', round(min_loss,3))

def visualise_model():
  sps = random.sample(xys_val, 4)
  sps_pred = get_tf_predictions(transformer, sps, PAD_IDX)
  sps_pred = tokenizer.decode_batch(sps_pred)
  sp = tokenizer.decode_batch(sps[0])
  print(sp[0].strip())
  print(sp[1].strip())
  print(sps_pred[0].strip())

def handle_training_interrupt():
  action = input('c -> continue\nh -> half learning rate\nx -> 10x learning rate\nq -> quit\ns -> save\nv -> visualize\n').strip()
  if action == 'c': pass
  elif action == 'h':
    for g in optimizer.param_groups:
      g['lr'] /= 2
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
    visualise_model()
  interrupt_handled()


EPOCHS = 100
prevent_interrupts()
for i in range(1, EPOCHS+1):
  train_loss = train_epoch(transformer, loss_fn, train_dataloader)
  eval_loss = evaluate(transformer, loss_fn, test_dataloader)
  print(f'Epoch {i} done. t: {round(train_loss,3)}, v: {round(eval_loss,3)}.',end=' ')

  save_model()

  if was_interrupted(): handle_training_interrupt()

# TODO use the masks properly
# TODO group into batches based on length
# TODO train on google colab