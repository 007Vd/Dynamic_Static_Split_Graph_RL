#%%
import torch
import torch.nn as nn
from src.rl.rmlp import RMLP

class Critic(nn.Module):
    def __init__(self,input_dim=1024,d_att=512,num_layers=4,dropout=0.1):
        super().__init__()
        self.rmlp=RMLP(
            input_dim=input_dim,
            d_att=d_att,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self,E_final):
        market_state=E_final.mean(dim=0,keepdim=True)
        value=self.rmlp(market_state)
        return value.squeeze()

