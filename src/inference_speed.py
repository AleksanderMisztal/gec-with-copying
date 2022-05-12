from timeit import default_timer as timer
from datetime import datetime
from copygec.models.optimizer import get_std_opt
from copygec.models.transformer_custom import make_model as TransformerCus
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
import copygec.gec_hub as hub

models = {
  'cus-1l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=1, device=hub.DEVICE),
  'cus-2l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=2, device=hub.DEVICE),
  'cus-3l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=3, device=hub.DEVICE),
  'cus-4l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=4, device=hub.DEVICE),
  'cus-5l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=5, device=hub.DEVICE),
  'cus-6l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=6, device=hub.DEVICE),

  'cop-1l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=1, copy=True, device=hub.DEVICE),
  'cop-2l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=2, copy=True, device=hub.DEVICE),
  'cop-3l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=3, copy=True, device=hub.DEVICE),
  'cop-4l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=4, copy=True, device=hub.DEVICE),
  'cop-5l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=5, copy=True, device=hub.DEVICE),
  'cop-6l':TransformerCus(VOCAB_S, PAD_IDX, num_layers=6, copy=True, device=hub.DEVICE),
}

xys = load_datasets()
print('Tracking inference times')
time = datetime.now().strftime("%m-%d_%Hh%Mm%Ss")
for name, model in models.items():
  model_name = f'{time}-{name}'
  print('# ' + model_name)
  opt = get_std_opt(model, model.d_model)
  start = timer()
  hub.train_model(model, xys['train'], xys['dev'], opt, 1, model_name, add_noise=True)
  end = timer()
  print(end - start)
  