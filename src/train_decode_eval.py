from datetime import datetime
from copygec.models.optimizer import get_std_opt
from copygec.models.transformer_custom import make_model as TransformerCus
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets, load_pretraining_sentences
from copygec.utils import unzip
import copygec.gec_hub as hub
import torch

num_layers = 3
time = datetime.now().strftime("%m-%d_%Hh%Mm%Ss")
model_name = f'{time}_{num_layers}l'

print('# ' + model_name)
transformer = TransformerCus(VOCAB_S, PAD_IDX, num_layers=num_layers, device=hub.DEVICE)
if torch.cuda.is_available(): transformer.cuda(device=0)

opt = get_std_opt(transformer, transformer.d_model)
sentences_it = load_pretraining_sentences() # loads in chunks of ~100k sentences
hub.synthetic_pretrain(transformer, sentences_it, opt)

xys = load_datasets()
print(len(xys['train']),len(xys['dev']),len(xys['test']))

print('Loaded data. Starting training...')
hub.train_model(transformer, xys['train'], xys['dev'], opt, 15, model_name, add_noise=True)

hub.load_model(transformer, model_name)
opt.renew()
transformer.generator.is_copying = True
hub.train_model(transformer, xys['train'], xys['dev'], opt, 15, model_name, add_noise=True)

# The model with smallest dev loss has been saved. The current model may have overfitted.
hub.load_model(transformer, model_name)

print('Predicting...')
orig, corr = unzip(xys['test'])
pred = hub.get_predictions(transformer, orig)
hub.save_results(orig, corr, pred, model_name)
hub.run_errant(model_name)

print('Testing copying...')
hub.visualise_copying(transformer, xys['test'], lim=3)
