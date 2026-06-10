#%%
import torch 
import torch.nn as nn
from mgan import StaticMGAN,DynamicMGAN
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
            dynamic_branch,static_branch
        ],dim=-1)


        return E_final

# %%
static_model=StaticMGAN()
dynamic_model=DynamicMGAN()

STATIC_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/dow30/static_graph.pt")
DYNAMIC_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/dow30/dynamic_graphs.pt")
x=torch.randn(29,128)
static_graph=torch.load(STATIC_DIR)
dynamic_graph=torch.load(DYNAMIC_DIR)
graph=dynamic_graph[0]

print("Static Branch")
E_s,attn=static_model(x,static_graph["edge_index"])
print(E_s.shape)
print("Dynamic Branch")
E_d,attn=dynamic_model(x,graph["edge_index"])
print(E_d.shape)

print("Fusion")
f_model=FusionLayer()
E_final=f_model(x,E_d,E_s)
print(E_final.shape)
# %%
