import torch
from pathlib import Path

from copygec.models.transformer_ref import Transformer as TransformerRef
from copygec.models.transformer_custom import make_model as TransformerCustom
from copygec.dataloader import DataLoader, load_datasets
from copygec.mytokenizer import PAD_IDX, to_token_idxs, tokenizer
from copygec.training import evaluate, train_epoch
import matplotlib.pyplot as plt

Path('./models/transformer').mkdir(parents=True, exist_ok=True)
def get_save_path(name): return './models/transformer/compare_2l_' + name + '.pt'
def get_loss_path(name): return './losses/compare_2l_' + name + '.png'

xys_train, xys_val = load_datasets('./data/')
xys_train = to_token_idxs(xys_train)
xys_val = to_token_idxs(xys_val)
print("Train / val set sizes:", len(xys_train), len(xys_val))

BATCH_SIZE = 128
LAYERS = 2
EPOCHS = 20
LEARNING_RATE = 0.001

train_dataloader = DataLoader(xys_train, BATCH_SIZE, PAD_IDX)
test_dataloader = DataLoader(xys_val, BATCH_SIZE, PAD_IDX)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device used:", DEVICE)

models = {
  'ref': TransformerRef(tokenizer.get_vocab_size(), PAD_IDX, num_layers=LAYERS, device=DEVICE),
  'custom': TransformerCustom(tokenizer.get_vocab_size(), PAD_IDX, num_layers=LAYERS, device=DEVICE, copy=False),
  'copy': TransformerCustom(tokenizer.get_vocab_size(), PAD_IDX, num_layers=LAYERS, device=DEVICE, copy=True)
}
optimizers = {name: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) for name, model in models.items()}
losses = {name: [] for name in models.keys()}

for name in models.keys():
  model = models[name]
  optimizer = optimizers[name]
  loss_history = losses[name]
  print(f'Training model "{name}"...')
  min_loss = evaluate(model, loss_fn, test_dataloader, DEVICE)
  loss_history.append(min_loss)
  for i in range(1, EPOCHS+1):
    train_loss = train_epoch(model, loss_fn, train_dataloader, optimizer, DEVICE, True)
    eval_loss = evaluate(model, loss_fn, test_dataloader, DEVICE)
    loss_history.append(eval_loss)
    print(f'Epoch {i} done. t: {round(train_loss,3)}, v: {round(eval_loss,3)}.',end=' ')
    if eval_loss < min_loss:
      min_loss = eval_loss
      torch.save(model.state_dict(), get_save_path(name))
      print('Saved!')
    else:
      print('Decreasing lr')
      for g in optimizer.param_groups: g['lr'] /= 2
  
for name, loss_history in losses.items():
  plt.plot(loss_history, label=name)
plt.savefig(get_loss_path(name), bbox_inches='tight')

