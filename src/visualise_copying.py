from copygec.models.transformer_custom import make_model as Transformer
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
import copygec.gec_hub as hub

num_layers = 1
model_name = '03-23_17:11:12_1l_15e_copy_noise'
transformer = Transformer(VOCAB_S, PAD_IDX, copy=True, num_layers=num_layers, device=hub.DEVICE)
# hub.load_model(transformer, model_name)
xys = load_datasets()['test']

hub.visualise_copying(transformer, xys)
