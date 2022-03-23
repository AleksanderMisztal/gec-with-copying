from torch.profiler import profile, record_function, ProfilerActivity
from copygec.models.transformer_custom import make_model as Transformer
from copygec.mytokenizer import VOCAB_S, PAD_IDX
from copygec.dataloader import load_datasets, sentences_to_padded_tensor
import copygec.gec_hub as hub


num_layers = 3
model_name = '3l_50e_copy_noise'
model = Transformer(VOCAB_S, PAD_IDX, copy=True, num_layers=num_layers, device=hub.DEVICE)
hub.load_model(model, model_name)
xys = load_datasets()['test']
x, y = xys[0]
src = sentences_to_padded_tensor([x])
tgt = sentences_to_padded_tensor([y])
tgt_in =  tgt[:-1, :]

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
  with record_function("model_inference"):
    out = model(src, tgt_in)

print(out.shape)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
prof.export_chrome_trace("trace.json")