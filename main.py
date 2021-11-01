import torch
from dataloader import get_orig_corr_pairs
from mytokenizer import to_token_idxs, tokenizer, PAD_IDX
from model import RefTransformer
from utils import count_params, to_padded_tensor, unzip

xys = get_orig_corr_pairs()
xys = to_token_idxs(xys)

transformer = RefTransformer(tokenizer.get_vocab_size())
transformer.load_state_dict(torch.load('./models/model.pt'))
print(count_params(transformer))

sentence_pairs = xys[:3]

xs, ys = unzip(xys)
src, tgt = to_padded_tensor(xs, PAD_IDX).T, to_padded_tensor(ys, PAD_IDX).T
out = transformer(src, tgt[:-1, :])
probs = torch.nn.Softmax(dim=2)(out)
words = torch.argmax(probs, dim=2)
kwords = torch.topk(probs, 8, dim=2).indices.transpose(0,1)
idxs = words.T.tolist()

def decode(idxs): return tokenizer.decode_batch(idxs, skip_special_tokens=False)

print(decode(kwords[:,:,0].tolist()))
print(decode(kwords[:,:,1].tolist()))
print(decode(kwords[:,:,2].tolist()))
print(decode(kwords[:,:,3].tolist()))
print(decode(kwords[:,:,4].tolist()))
print(decode(kwords[:,:,5].tolist()))
print(decode(kwords[:,:,6].tolist()))
print(decode(kwords[:,:,7].tolist()))