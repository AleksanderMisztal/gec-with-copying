from dataloader import load_gec_data, DataLoader
from model import RefTransformer
import torch
import bpe

def make_data_loader():
  xys = load_gec_data(short=True)

  tokenizer = bpe.load_tokenizer()
  pad_token, bos_token, eos_token = '<pad>', '<bos>', '<eos>'
  tokenizer.add_special_tokens([pad_token, bos_token, eos_token])
  PAD_IDX, BOS_IDX, EOS_IDX = tokenizer.token_to_id(pad_token), tokenizer.token_to_id(bos_token), tokenizer.token_to_id(eos_token)

  
  def enc(x): return tokenizer.encode(x).ids

  BATCH_SIZE = 4

  xys = [(enc(x), [BOS_IDX]+enc(y)) for x, y in xys]
  dl = DataLoader(xys, BATCH_SIZE, PAD_IDX)

  return dl

  # print(PAD_IDX, BOS_IDX, EOS_IDX)

# enc = tokenizer.encode('<bos>Hello, this is an example sentence created by Aleksander.<pad><pad>')

# print(enc.tokens)
# print(enc.ids)
# print(tokenizer.get_vocab_size())


# model = RefTransformer(tokenizer.get_vocab_size())
# x, y = xys[0]
# x = torch.tensor([tokenizer.encode(x).ids]).T
# y = torch.tensor([tokenizer.encode(y).ids]).T
# print('x, y shapes:', x.shape, y.shape)
# y_pred = model(x, y)
# print('pred shape:', y_pred.shape, y_pred)

dl = make_data_loader()

for xs, ys in dl:
  print(xs)
  print(ys)
  input()

# https://pytorch.org/tutorials/beginner/translation_transformer.html