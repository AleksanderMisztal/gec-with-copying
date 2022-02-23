import torch
from pathlib import Path

from copygec.models.transformer_ref import Transformer as TransformerRef
from copygec.models.transformer_custom import make_model as TransformerCustom
from copygec.dataloader import DataLoader, load_datasets
from copygec.mytokenizer import PAD_IDX, to_token_idxs
from copygec.utils import to_padded_tensor

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
  'ref': TransformerRef(10000, PAD_IDX, num_layers=LAYERS, device=DEVICE),
  'custom': TransformerCustom(10000, PAD_IDX, num_layers=LAYERS, device=DEVICE, copy=False),
  'copy': TransformerCustom(10000, PAD_IDX, num_layers=LAYERS, device=DEVICE, copy=True)
}
optimizers = {name: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) for name, model in models.items()}

xs = to_padded_tensor([[0, 12, 34, 99]], PAD_IDX).T
ys = to_padded_tensor([[0, 12, 34, 99]], PAD_IDX).T

logits = {name: model(xs, ys[:-1]).flatten().detach().numpy() for name, model in models.items()}

import matplotlib.pyplot as plt
for name, ls in logits.items():
  print(ls)
  plt.hist(ls)
  plt.show()
