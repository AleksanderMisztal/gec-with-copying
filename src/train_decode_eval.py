from copygec.models.transformer_ref import Transformer
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
from copygec.utils import unzip
import copygec.gec_hub as hub

model_name = 'test'#'ref_3l_10e'
num_layers = 1
epochs = 1

transformer = Transformer(VOCAB_S, PAD_IDX, num_layers=num_layers, device=hub.DEVICE)
xys = load_datasets()

print('Loaded data. Starting training...')
hub.train_model(transformer, xys['train'][:16], xys['dev'][:16], epochs, model_name)
hub.load_model(transformer, model_name)

print('Predicting...')
orig, corr = unzip(xys['test'][:16])
pred = hub.get_predictions(transformer, orig)
hub.save_results(orig, corr, pred, model_name)
print('Running errant...')
hub.run_errant(model_name)
