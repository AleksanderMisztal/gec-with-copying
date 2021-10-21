import random

orig1 = open('./data/conll.orig.txt').readlines()
corr1 = open('./data/conll.corr.txt').readlines()

orig2 = open('./data/nucle.orig.txt').readlines()
corr2 = open('./data/nucle.corr.txt').readlines()

orig = orig1 + orig2
corr = corr1 + corr2

orig = [s.strip() for s in orig]
corr = [s.strip() for s in corr]

xys = list(zip(orig, corr))
xys = [(x, y) for x, y in xys if x != y]

random.shuffle(xys)

print(len(xys))
print(sum(len(x.split()) for x, y in xys)/len(xys))
print(sum(len(y.split()) for x, y in xys)/len(xys))

import bpe

#tokenizer = bpe.train_tokenizer([s for xy in xys for s in xy])
tokenizer = bpe.load_tokenizer()
enc = tokenizer.encode('Hello, this is an example sentence created by aleksander.')

print(enc.tokens)
print(enc.ids)

# for x, y in xys:
#   print(x)
#   print(y)
#   input()
