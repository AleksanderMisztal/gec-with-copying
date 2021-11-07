import json
import random

orig1 = open('./data/conll.orig.txt').readlines()
corr1 = open('./data/conll.corr.txt').readlines()

orig2 = open('./data/nucle.orig.txt').readlines()
corr2 = open('./data/nucle.corr.txt').readlines()

orig = [s.strip() for s in orig1 + orig2]
corr = [s.strip() for s in corr1 + corr2]

xys = list(zip(orig, corr))
xys = [(x, y) for x, y in xys if x != y]

random.shuffle(xys)

with open('./data/train.json', 'w', encoding='utf-8') as f:
    json.dump(xys[1024:], f, ensure_ascii=False, indent=4)
with open('./data/val.json', 'w', encoding='utf-8') as f:
    json.dump(xys[:1024], f, ensure_ascii=False, indent=4)