import torch
from copygec.dataloader import DataLoader, load_datasets
from copygec.mytokenizer import PAD_IDX, tokenizer, to_token_idxs
from src.copygec.models.transformer_custom import Transformer
from src.copygec.training import train_epoch

xys_train, xys_val = load_datasets('./data/')
xys_train = to_token_idxs(xys_train)
xys_val = to_token_idxs(xys_val)

train_dataloader = DataLoader(xys_train, 128, PAD_IDX)
val_dataloader = DataLoader(xys_val, 32, PAD_IDX)
print("Data loaded! Train / val set sizes:",len(xys_train), len(xys_val))

transformer = Transformer(tokenizer.get_vocab_size(), 512, 2, PAD_IDX)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)

train_epoch(transformer, loss_fn, train_dataloader, optimizer, torch.device('cpu'), True)