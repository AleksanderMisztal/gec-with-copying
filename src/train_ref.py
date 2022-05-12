from datetime import datetime
from copygec.models.transformer_ref import Transformer as TransformerRef
from copygec.models.optimizer import get_std_opt
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
from copygec.utils import unzip
import copygec.gec_hub as hub

models = {
  'ref-3l':TransformerRef(VOCAB_S, PAD_IDX, num_layers=3, device=hub.DEVICE),
  'ref-4l':TransformerRef(VOCAB_S, PAD_IDX, num_layers=4, device=hub.DEVICE),
}

xys = load_datasets()
print(len(xys['train']),len(xys['dev']),len(xys['test']))
time = datetime.now().strftime("%m-%d_%Hh%Mm%Ss")
for name, model in models.items():
  model_name = f'{time}-{name}'
  print('# ' + model_name)
  opt = get_std_opt(model, model.d_model)
  hub.train_model(model, xys['train'], xys['dev'], opt, 10, model_name, add_noise=True)
  hub.load_model(model, model_name)
  print('Predicting...')
  orig, corr = unzip(xys['test'])
  pred = hub.get_predictions(model, orig)
  hub.save_results(orig, corr, pred, model_name)
  hub.run_errant(model_name)
  print('done', flush=True)
