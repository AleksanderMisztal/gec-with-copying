import torch

class NoamOpt:
  def __init__(self, d_model, factor, warmup, optimizer):
    self.d_model = d_model
    self.factor = factor
    self.warmup = warmup
    self.optimizer = optimizer
    self.cp = 0
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
    if step < self.cp + self.warmup:
      return self.rate(self.cp+self.warmup) * (step-self.cp) / self.warmup
    else:
      return self.factor * (self.d_model ** -.5) * (step ** -.5)

  
  def renew(self):
    self.cp = self._step

        
def get_std_opt(model=None, d_model=512):
  if model is None: opt = None
  else: opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
  return NoamOpt(d_model, 1, 4000, opt)
