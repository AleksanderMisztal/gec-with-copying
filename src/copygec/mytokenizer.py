from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from pathlib import Path

from src.copygec.dataloader import load_datasets

TRAIN_TOKENIZER = False
TOKENIZER_PATH = './models/tokenizer'
 
VOCAB_S = 30000

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

def enc(x): return tokenizer.encode(x).ids
def to_token_idxs(xys): return [(enc(x), [BOS_IDX]+enc(y)+[EOS_IDX]) for x, y in xys]

if __name__ == '__main__':
  trainer = BpeTrainer(vocab_size=VOCAB_S, show_progress=True, initial_alphabet=ByteLevel.alphabet())
  train, _ = load_datasets('./data/')
  sentences = [s for xy in train for s in xy]
  tokenizer.train_from_iterator(sentences, trainer)
  Path(TOKENIZER_PATH).mkdir(parents=True, exist_ok=True)
  tokenizer.model.save(TOKENIZER_PATH)
  print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
else:
  tokenizer.model = BPE.from_file(TOKENIZER_PATH + '/vocab.json', TOKENIZER_PATH + '/merges.txt')

PAD_T, BOS_T, EOS_T = '<pad>', '<bos>', '<eos>'
tokenizer.add_special_tokens([PAD_T, BOS_T, EOS_T])
PAD_IDX, BOS_IDX, EOS_IDX = tokenizer.token_to_id(PAD_T), tokenizer.token_to_id(BOS_T), tokenizer.token_to_id(EOS_T)
