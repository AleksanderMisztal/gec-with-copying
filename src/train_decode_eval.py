import torch
from pathlib import Path
#from copygec.models.transformer_ref import Transformer
from copygec.models.transformer_custom import make_model as Transformer
from copygec.dataloader import DataLoader, load_datasets
from copygec.mytokenizer import PAD_IDX, to_token_idxs, tokenizer
from copygec.training import evaluate, train_epoch
from copygec.models.optimizer import get_std_opt
from evaluate_model import write_for_evaluation
from run_errant import run_errant

def train_model(model, xys, optimizer, batch_size, epochs, device, model_name):
  MODEL_SAVE_PATH = './models/transformer/' + model_name + '.pt'
  Path('./models/transformer').mkdir(parents=True, exist_ok=True)

  xys_train, xys_val = xys
  xys_train = to_token_idxs(xys_train)
  xys_val = to_token_idxs(xys_val)

  print("Train / val set sizes:", len(xys_train), len(xys_val))
  print("Device used:", device)

  train_dataloader = DataLoader(xys_train, batch_size, PAD_IDX)
  test_dataloader = DataLoader(xys_val, batch_size, PAD_IDX)

  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

  min_loss = evaluate(model, loss_fn, test_dataloader, device)
  print(f'Initial validation loss: {round(min_loss, 3)}.')
  for i in range(1, epochs+1):
    train_loss = train_epoch(model, loss_fn, train_dataloader, optimizer, device, True)
    eval_loss = evaluate(model, loss_fn, test_dataloader, device)
    print(f'Epoch {i} done. t: {round(train_loss,3)}, v: {round(eval_loss,3)}.',end=' ')
    if eval_loss < min_loss:
      min_loss = eval_loss
      torch.save(model.state_dict(), MODEL_SAVE_PATH)
      print('Saved!')
    else: print()

SAVENAME = 'exp1'
N_LAYERS = 2
BATCH_SIZE = 128
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer = Transformer(tokenizer.get_vocab_size(), PAD_IDX, num_layers=N_LAYERS, device=DEVICE)
optimizer = get_std_opt(transformer, transformer.d_model)

xys = load_datasets('./data')

print('Starting training...')
train_model(transformer, xys, optimizer, BATCH_SIZE, EPOCHS, DEVICE, SAVENAME)

print('Training done! Starting to write sentences...')
write_for_evaluation(transformer, xys[1], SAVENAME)

print('Writing done! Starting to evaluate with errant...')
run_errant(f'./out/{SAVENAME}')