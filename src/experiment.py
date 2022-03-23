import torch
from copygec.utils import zip_tensors

a = torch.tensor([float('-inf'),float('-inf'),1,2,3,4])
print(torch.softmax(a, dim=0))