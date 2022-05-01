import os

def printdir(dir, depth=1):
  print(f".{depth} \dir"+ "{"+dir.split('/')[-1]+"}.", sep='')
  if dir in ['./venv', './.git', './out', './models', './logs']: return
  
  dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
  for d in dirs:
    printdir(os.path.join(dir, d), depth+1)


  files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
  for f in files:
    print(f".{depth+1} {f}.", sep='')
  

printdir('.')