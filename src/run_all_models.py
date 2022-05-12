from datetime import datetime
from copygec.models.transformer_ref import Transformer as TransformerRef
from copygec.models.optimizer import get_std_opt
from copygec.models.transformer_custom import make_model as TransformerCus
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
from copygec.utils import unzip
import copygec.gec_hub as hub

models = {
  # 'ref-1l':TransformerRef(VOCAB_S, PAD_IDX, num_layers=1, device=hub.DEVICE),

  # 'ref-2l':TransformerRef(VOCAB_S, PAD_IDX, num_layers=2, device=hub.DEVICE),
  # 'cus-2l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=2, device=hub.DEVICE),
  # 'cop-2l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=2, copy=True, device=hub.DEVICE),

  # 'ref-3l':TransformerRef(VOCAB_S, PAD_IDX, num_layers=3, device=hub.DEVICE),
  # 'cus-3l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=3, device=hub.DEVICE),
  # 'cop-3l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=3, copy=True, device=hub.DEVICE),
  
  # 'ref-4l':TransformerRef(VOCAB_S, PAD_IDX, num_layers=4, device=hub.DEVICE),
  # 'cus-4l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=4, device=hub.DEVICE),
  # 'cop-4l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=4, copy=True, device=hub.DEVICE),

  'ref-5l':TransformerRef(VOCAB_S, PAD_IDX, num_layers=5, device=hub.DEVICE),
  # 'cus-5l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=5, device=hub.DEVICE),
  # 'cop-5l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=5, copy=True, device=hub.DEVICE),
  
  'ref-6l':TransformerRef(VOCAB_S, PAD_IDX, num_layers=6, device=hub.DEVICE),
}

xys = load_datasets()
print(len(xys['train']),len(xys['dev']),len(xys['test']))
time = datetime.now().strftime("%m-%d_%Hh%Mm%Ss")
for name, model in models.items():
  model_name = f'{time}-{name}'
  print('# ' + model_name)
  opt = get_std_opt(model, model.d_model)
  hub.train_model(model, xys['train'], xys['dev'], opt, 10, model_name, add_noise=False)
  hub.load_model(model, model_name)
  print('Predicting...')
  orig, corr = unzip(xys['test'])
  pred = hub.get_predictions(model, orig)
  hub.save_results(orig, corr, pred, model_name)
  hub.run_errant(model_name)
  print('done', flush=True)
