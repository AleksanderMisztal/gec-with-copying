from copygec.models.transformer_custom import make_model as Transformer
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets
import copygec.gec_hub as hub

num_layers = 3
model_name = '3l_50e_copy_noise'
transformer = Transformer(VOCAB_S, PAD_IDX, copy=True, num_layers=num_layers, device=hub.DEVICE)
hub.load_model(transformer, model_name)
xys = load_datasets()['test']
print("Visualising")
hub.visualise_copying(transformer, xys)

