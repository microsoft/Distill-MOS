import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import Config


class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    Source: https://github.com/gabrielmittag/NISQA/blob/ac831378483aa75876c1147acb04b104f6f1d10c/nisqa/NISQA_lib.py#L1156-L1183
    '''         
    def __init__(self, config: Config):
        super().__init__()
        
        self.linear1 = nn.Linear(config.dim_head_in, 2*config.dim_head_in)
        self.linear2 = nn.Linear(2*config.dim_head_in, 1)
        
        self.linear3 = nn.Linear(config.dim_head_in, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x  
