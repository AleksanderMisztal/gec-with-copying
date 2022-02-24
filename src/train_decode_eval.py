import torch
from copygec.models.transformer_ref import Transformer
# from copygec.models.transformer_custom import make_model as Transformer
from copygec.dataloader import load_datasets
from copygec.mytokenizer import PAD_IDX, tokenizer
from copygec.models.optimizer import get_std_opt
from copygec.training import train_model, run_errant, write_for_evaluation

SAVENAME = 'exp1'
LOADNAME = './models/transformer/' + SAVENAME + '.pt'
N_LAYERS = 3
BATCH_SIZE = 128
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer = Transformer(tokenizer.get_vocab_size(), PAD_IDX, num_layers=N_LAYERS, device=DEVICE)
optimizer = get_std_opt(transformer, transformer.d_model)

xys_train, xys_dev, xys_test = load_datasets()

#train_model(transformer, xys_train, xys_dev, optimizer, BATCH_SIZE, EPOCHS, DEVICE, SAVENAME)
transformer.load_state_dict(torch.load(LOADNAME))
write_for_evaluation(transformer, xys_test, BATCH_SIZE, DEVICE, SAVENAME)
run_errant(f'./out/{SAVENAME}')

# TODO Get a high score with the ref transformer
  # TODO Clean the data? Try tokenizing Hello.friends
# TODO Get a high score with the custom transformer
# TODO Get a high score with the copying transformer
  # TODO Debug and test copying
  # TODO Masking in copying
  # TODO How to combine the scores?
  # TODO Add pad masks to custom
# TODO Create masks outside the model
# TODO Label smoothing as regularization?
# TODO Make beam search work
# TODO Visualise attention
# TODO Implement qualitative visualisations