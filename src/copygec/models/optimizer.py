import torch

class NoamOpt:
  def __init__(self, d_model, factor, warmup, optimizer):
    self.d_model = d_model
    self.factor = factor
    self.warmup = warmup
    self.optimizer = optimizer
    self._step = 0
    self._rate = 0
      
  def step(self):
    self._step += 1
    rate = self.rate()
    for p in self.optimizer.param_groups:
      p['lr'] = rate
    self._rate = rate
    self.optimizer.step()
  
  def zero_grad(self):
    self.optimizer.zero_grad()
      
  def rate(self, step = None):
    if step is None: step = self._step
    return self.factor * \
      (self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model, d_model):
  return NoamOpt(d_model, 0.25, 200,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))