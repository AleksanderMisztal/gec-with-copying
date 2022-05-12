from copygec.dataloader import load_datasets
from copygec.mytokenizer import train_tokenizer

xys = load_datasets()['train']
ys = [y for x,y in xys]
train_tokenizer(ys)
