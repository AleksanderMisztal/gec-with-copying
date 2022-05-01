from copygec.models.transformer_custom import make_model as Transformer
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
from copygec.utils import unzip
import copygec.gec_hub as hub

num_layers = 3
model_name = '04-04_10h39m01s_3l_30e_copy_noise'

transformer = Transformer(VOCAB_S, PAD_IDX, copy=True, num_layers=num_layers, device=hub.DEVICE)
hub.load_model(transformer, model_name)

# xys = load_datasets(dummy=True)

# print('Predicting...')
# orig, corr = unzip(xys['test'])
# pred = hub.get_bs_predictions(transformer, orig)
# hub.save_results(orig, corr, pred, model_name)
# hub.run_errant(model_name)

# print('Testing copying...')
# hub.visualise_copying(transformer, xys['test'], lim=3)


pred = hub.get_bs_predictions(transformer, ['Cats hates water.'])

print(pred)