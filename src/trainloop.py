import torch
from pathlib import Path
from argparse import ArgumentParser

#from copygec.models.transformer_ref import Transformer
from copygec.models.transformer_custom import make_model as Transformer
from copygec.dataloader import DataLoader, load_datasets
from copygec.mytokenizer import PAD_IDX, to_token_idxs, tokenizer
from copygec.training import evaluate, train_epoch
from copygec.models.optimizer import get_std_opt

parser = ArgumentParser()
parser.add_argument("--ln", dest="loadname", help="Load model with this name")
parser.add_argument("--sn", dest="savename", help="Save with this name", required=True)
parser.add_argument("--epochs", dest="epochs", type=int, help="Run for this many epochs", required=True)
parser.add_argument("--layers", dest="layers", type=int, help="Create this many layers in the transformer", required=True)
parser.add_argument("--lr", dest="learningrate", type=float, help="Learning rate", default=0.001)
args = parser.parse_args()

IS_MODEL_LOADED = args.loadname is not None
if IS_MODEL_LOADED: MODEL_LOAD_PATH = './models/transformer/' + args.loadname + '.pt'
MODEL_SAVE_PATH = './models/transformer/' + args.savename + '.pt'

Path('./models/transformer').mkdir(parents=True, exist_ok=True)

if IS_MODEL_LOADED and not Path(MODEL_LOAD_PATH).exists(): 
  print('Attmepting to load a model that does not exist!')
  exit()

xys_train, xys_val = load_datasets('./data/')
xys_train = to_token_idxs(xys_train)
xys_val = to_token_idxs(xys_val)
print("Train / val set sizes:", len(xys_train), len(xys_val))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = args.learningrate
BATCH_SIZE = 128
train_dataloader = DataLoader(xys_train, BATCH_SIZE, PAD_IDX)
test_dataloader = DataLoader(xys_val, BATCH_SIZE, PAD_IDX)

transformer = Transformer(tokenizer.get_vocab_size(), PAD_IDX, num_layers=args.layers, device=DEVICE)
if IS_MODEL_LOADED: transformer.load_state_dict(torch.load(MODEL_LOAD_PATH))
print("Device used:", DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = get_std_opt(transformer, transformer.d_model)

min_loss = evaluate(transformer, loss_fn, test_dataloader, DEVICE)
print(f'Initial validation loss: {round(min_loss, 3)}.')
for i in range(1, args.epochs+1):
  train_loss = train_epoch(transformer, loss_fn, train_dataloader, optimizer, DEVICE, True)
  eval_loss = evaluate(transformer, loss_fn, test_dataloader, DEVICE)
  print(f'Epoch {i} done. t: {round(train_loss,3)}, v: {round(eval_loss,3)}.',end=' ')
  if eval_loss < min_loss:
    min_loss = eval_loss
    torch.save(transformer.state_dict(), MODEL_SAVE_PATH)
    print('Saved!')
  else: print()

# TODO Add pad masks
# TODO Test masking in custom
# TODO Train / Just decode again with the masking added to decoding
# TODO Label smoothing
# TODO Implement copying
# TODO Make beam search work
# TODO Visualise attention
# TODO Implement qualitative visualisations