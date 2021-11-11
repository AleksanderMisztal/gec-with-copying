import json
import random
from copygec.utils import noise
from copygec.dataloader import load_datasets
from copygec.mytokenizer import enc, tokenizer

dec = tokenizer.decode
xys_train, xys_val = load_datasets('./data/')
vs = tokenizer.get_vocab_size()-3
encs = [enc(y) for x, y in xys_train]
xys = [(dec(noise(y, vs)[0]), dec(y)) for y in encs]

random.shuffle(xys)

path = './data-synt/'
val_s = 1024
with open(path+'train.json', 'w', encoding='utf-8') as f:
    json.dump(xys[val_s:], f, ensure_ascii=False, indent=4)
with open(path+'val.json', 'w', encoding='utf-8') as f:
    json.dump(xys[:val_s], f, ensure_ascii=False, indent=4)