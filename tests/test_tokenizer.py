import random
from copygec.mytokenizer import enc, dec
from copygec.dataloader import load_datasets

xys = load_datasets()['train']
random.shuffle(xys)
sentences = [s for xy in xys[:100] for s in xy]
sentences += [
  "I'm do n't don't a1b2c3... ,., ! abc.def 123.456 . . !",
  "9.30pm"
]
failed = False
for s in sentences:
  t = dec(enc(s))
  if not s == t:
    print(f'Test failed!\n{s}\n{t}\n')
    failed = True 
if not failed: print('All tests passed!')