from datetime import datetime
from copygec.models.optimizer import get_std_opt
from copygec.models.transformer_custom import make_model as TransformerCus
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
from copygec.utils import unzip
import copygec.gec_hub as hub

models = {
  'cop-2l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=2, copy=False, device=hub.DEVICE),
  'cop-3l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=3, copy=False, device=hub.DEVICE),
  'cop-4l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=4, copy=False, device=hub.DEVICE),
  'cop-5l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=5, copy=False, device=hub.DEVICE),
}

xys = load_datasets()
time = datetime.now().strftime("%m-%d_%Hh%Mm%Ss")
for name, model in models.items():
  model_name = f'{time}-{name}'
  print('# ' + model_name, flush=True)
  opt = get_std_opt(model, model.d_model)
  hub.train_model(model, xys['train'], xys['dev'], opt, 10, model_name, add_noise=True)
  hub.load_model(model, model_name)
  model.generator.is_copying = True
  opt.renew()
  hub.train_model(model, xys['train'], xys['dev'], opt, 10, model_name, add_noise=True)
  hub.load_model(model, model_name)
  orig, corr = unzip(xys['test'])
  pred = hub.get_predictions(model, orig)
  hub.save_results(orig, corr, pred, model_name)
  hub.run_errant(model_name)
