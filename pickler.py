import pickle

def save(title, data):
  f = open('./pickles/' + title + '.pickle','wb')
  pickle.dump(data,f)
  f.close()

def load(title):
  f = open('./pickles/' + title + '.pickle', 'rb')
  data = pickle.load(f)
  f.close()
  return data