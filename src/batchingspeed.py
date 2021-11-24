import torch
from copygec.dataloader import DataLoader, load_datasets
from copygec.model import RefTransformer
from copygec.mytokenizer import PAD_IDX, to_token_idxs, tokenizer
from timeit import default_timer as timer

MODEL_LOAD_NAME = 'transformer/model_eos'
model = RefTransformer(tokenizer.get_vocab_size())
model.load_state_dict(torch.load('./models/' + MODEL_LOAD_NAME + '.pt'))

xys_train, xys_val = load_datasets('./data/')

xys_train = to_token_idxs(xys_train)

BATCH_SIZE = 128
TEST_SIZE = 100

batch_loader = iter(DataLoader(xys_train, BATCH_SIZE, PAD_IDX))
single_loader = iter(DataLoader(xys_train, 1, PAD_IDX))

src, tgt = next(single_loader)
tgt_input = tgt[:-1,:]
torch.manual_seed(0)
l1 = model(src, tgt_input)
torch.manual_seed(0)
l2 = model(src, tgt_input)
l1 = torch.reshape(l1, (-1,))
l2 = torch.reshape(l2, (-1,))
print(l1[0], l2[0])
print(torch.norm(l1-l2))

print("Timing batching...")
start = timer()
for i in range(TEST_SIZE):
  src, tgt = next(batch_loader)
  tgt_input = tgt[:-1, :]
  logits = model(src, tgt_input)
end = timer()
print(round(end - start,3))

print("Timing single...")
start = timer()
for i in range(TEST_SIZE*BATCH_SIZE):
  src, tgt = next(single_loader)
  tgt_input = tgt[:-1, :]
  logits = model(src, tgt_input)
end = timer()
print(round(end - start,3))