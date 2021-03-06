from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Whitespace, Digits
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

TOKENIZER_PATH = './models/tokenizer'
VOCAB_S = 30000
PAD_T, BOS_T, EOS_T, MASK_T = '<pad>', '<bos>', '<eos>', '<mask>'

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()
tokenizer.post_processor = TemplateProcessing(
    single=f"{BOS_T} $A {EOS_T}",
    special_tokens=[(BOS_T, 0), (EOS_T, 1)],
)
tokenizer.model = BPE.from_file(TOKENIZER_PATH + '/vocab.json', TOKENIZER_PATH + '/merges.txt')
tokenizer.add_special_tokens([BOS_T, EOS_T, PAD_T, MASK_T])
PAD_IDX, BOS_IDX, EOS_IDX, MASK_IDX = tokenizer.token_to_id(PAD_T), tokenizer.token_to_id(BOS_T), tokenizer.token_to_id(EOS_T), tokenizer.token_to_id(MASK_T)

def enc(x): return tokenizer.encode(x).ids
def dec(y, skip_special=True): return " ".join(tokenizer.decode(y, skip_special_tokens=skip_special).split())
def ids_to_tokens(ids): return [tokenizer.id_to_token(id) for id in ids]
def id_to_token(id): return tokenizer.id_to_token(id)

def train_tokenizer(sentences):
  trainer = BpeTrainer(
    vocab_size=VOCAB_S,
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens=[BOS_T, EOS_T, PAD_T, MASK_T],
    show_progress=True,
  )
  tokenizer.train_from_iterator(sentences, trainer)
  Path(TOKENIZER_PATH).mkdir(parents=True, exist_ok=True)
  tokenizer.model.save(TOKENIZER_PATH)

