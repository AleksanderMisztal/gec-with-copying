from datetime import datetime
from copygec.models.transformer_custom import make_model as Transformer
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets, load_pretraining_sentences
from copygec.utils import unzip
import copygec.gec_hub as hub
import torch

num_layers = 3
epochs = 30
dummy = False
noise = False
copy = False

dummy_s = '_dummy' if dummy else ''
copy_s = '_copy' if copy else ''
noise_s = '_noise' if noise else ''
time = datetime.now().strftime("%m-%d_%Hh%Mm%Ss")

model_name = f'{time}_{num_layers}l_{epochs}e{dummy_s}{copy_s}{noise_s}'

print('# ' + model_name)
transformer = Transformer(VOCAB_S, PAD_IDX, copy=copy, num_layers=num_layers, device=hub.DEVICE)
if torch.cuda.is_available(): transformer.cuda(device=0)

#print('Pretraining...')
#transformer.generator.is_copying = False
# Loads pretraining sentences in chunks of 100_000
#sentences_it = load_pretraining_sentences('./data/news.txt')
# sentences_it = [sentences[:5000] for sentences in sentences_it]
#hub.synthetic_pretrain(transformer, sentences_it)

xys = load_datasets(dummy=dummy)
print(len(xys['train']),len(xys['dev']),len(xys['test']))

print('Loaded data. Starting training...')
#transformer.generator.is_copying = True
hub.train_model(transformer, xys['train'], xys['dev'], epochs, model_name, add_noise=noise)

print('Predicting...')
orig, corr = unzip(xys['test'])
pred = hub.get_predictions(transformer, orig)
hub.save_results(orig, corr, pred, model_name)
hub.run_errant(model_name)

print('Testing copying...')
hub.visualise_copying(transformer, xys['test'], lim=3)