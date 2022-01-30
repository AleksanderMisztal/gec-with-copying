import torch
from copygec.dataloader import DataLoader, load_datasets
from copygec.decoding import beam_search_decode, greedy_decode
from copygec.mytokenizer import PAD_IDX, tokenizer
from copygec.models.transformer_custom import Transformer
from copygec.training import train_epoch

xys_train, xys_val = load_datasets('./data/')

train_dataloader = DataLoader(xys_train, 128, PAD_IDX)
val_dataloader = DataLoader(xys_val, 32, PAD_IDX)
print("Data loaded! Train / val set sizes:",len(xys_train), len(xys_val))

LOADNAME = 'custom1l'
transformer = Transformer(tokenizer.get_vocab_size(), PAD_IDX, num_layers=1)
transformer.load_state_dict(torch.load('./models/transformer/' + LOADNAME + '.pt'))

print('All loaded, decoding...')

out = greedy_decode(transformer, xys_val[0][0])
outs = beam_search_decode(transformer, xys_val[0][0])
print(xys_val[0])
print()
print(outs)
print()
print(out)
