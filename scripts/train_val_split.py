import json

train_orig = open('./data/train.orig.txt', encoding='utf-8').readlines()
train_corr = open('./data/train.corr.txt', encoding='utf-8').readlines()

dev_orig = open('./data/dev.orig.txt', encoding='utf-8').readlines()
dev_corr = open('./data/dev.corr.txt', encoding='utf-8').readlines()

xys_train = [(x.strip(), y.strip()) for x, y in zip(train_orig, train_corr) if x != y]
xys_dev = [(x.strip(), y.strip()) for x, y in zip(dev_orig, dev_corr) if x != y]


with open('./data/train.json', 'w', encoding='utf-8') as f:
    json.dump(xys_train, f, ensure_ascii=False, indent=4)
with open('./data/val.json', 'w', encoding='utf-8') as f:
    json.dump(xys_dev, f, ensure_ascii=False, indent=4)