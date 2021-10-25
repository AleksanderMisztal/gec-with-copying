import torch
import bpe
from model import RefTransformer
from dataloader import DataLoader

PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2 # TODO set to correct idxs
special_symbols = ['<pad>', '<bos>', '<eos>']

tokenizer = bpe.load_tokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size()
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = RefTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, VOCAB_SIZE, VOCAB_SIZE, FFN_HID_DIM)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

train_dataloader = DataLoader(xys, BATCH_SIZE, PAD_IDX)


def create_mask(src, tgt_in):
  t_len = tgt_in.shape[1]

  tgt_mask = torch.triu(torch.ones(t_len, t_len), diagonal=1) * float('-inf')

  src_padding_mask = (src == PAD_IDX) 
  tgt_padding_mask = (tgt_in == PAD_IDX)

  return tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, loss_fn, train_dataloader):
    model.train()
    losses = 0

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)