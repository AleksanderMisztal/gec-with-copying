from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

VOCAB_S = 30000

def make_tokenizer():
  tokenizer = Tokenizer(BPE())
  tokenizer.pre_tokenizer = ByteLevel()
  tokenizer.decoder = ByteLevelDecoder()
  return tokenizer

def train_tokenizer(sentences):
  tokenizer = make_tokenizer()
  trainer = BpeTrainer(vocab_size=VOCAB_S, show_progress=True, initial_alphabet=ByteLevel.alphabet())
  tokenizer.train_from_iterator(sentences, trainer)

  tokenizer.model.save('./tokenizer/')

  print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
  return tokenizer

def load_tokenizer(path='./tokenizer'):
  tokenizer = make_tokenizer()
  tokenizer.model = BPE.from_file(path + '/vocab.json', path + '/merges.txt')
  return tokenizer


#tokenizer = bpe.train_tokenizer([s for xy in xys for s in xy])
