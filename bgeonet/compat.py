import torch
from torch import nn
from torch.nn.init import trunc_normal_
def to_2tuple(x): return x if isinstance(x,tuple) else (x,x)
class DropPath(nn.Module):
 def __init__(self,drop_prob=0.): super().__init__();self.drop_prob=drop_prob
 def forward(self,x):
  if not self.training or not self.drop_prob:return x
  keep=1-self.drop_prob;shape=(x.shape[0],)+(1,)*(x.ndim-1)
  return x*x.new_empty(shape).bernoulli_(keep)/keep
