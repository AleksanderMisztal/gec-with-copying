# from copygec.models.transformer_ref import Transformer
from copygec.models.transformer_custom import make_model as Transformer
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
from copygec.utils import unzip
import copygec.gec_hub as hub

num_layers = 2
epochs = 20
dummy = False
noise = True
copy = True

dummy_s = '_dummy' if dummy else ''
copy_s = '_copy' if copy else ''
noise_s = '_noise' if copy else ''

model_name = f'{num_layers}l_{epochs}e{dummy_s}{copy_s}{noise_s}'

print('# ' + model_name)
transformer = Transformer(VOCAB_S, PAD_IDX, copy=copy, num_layers=num_layers, device=hub.DEVICE)
transformer.cuda(device=0)
xys = load_datasets(dummy=dummy)

print('Loaded data. Starting training...')
hub.train_model(transformer, xys['train'], xys['dev'], epochs, model_name, add_noise=noise)

print('Predicting...')
orig, corr = unzip(xys['test'])
pred = hub.get_predictions(transformer, orig)
hub.save_results(orig, corr, pred, model_name)
hub.run_errant(model_name)
