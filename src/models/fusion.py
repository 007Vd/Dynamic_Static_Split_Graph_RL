#%%
import torch 
import torch.nn as nn
from src.models.mgan import StaticMGAN,DynamicMGAN
from pathlib import Path
class FusionLayer(nn.Module):
    def __init__(self,hidden_dim=128,d_att=512):
        super().__init__()
        self.W_ori=nn.Linear(hidden_dim,d_att)

    def forward(self,E_l,E_d,E_s):
        E_ori=self.W_ori(E_l)
        dynamic_branch=E_d+E_ori
        static_branch=E_s+E_ori

        E_final= torch.cat([
            dynamic_branch,static_branch],dim=-1)


        return E_final


