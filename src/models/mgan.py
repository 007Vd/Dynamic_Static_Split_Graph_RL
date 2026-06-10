#%%
import torch
import torch.nn as nn
from tga_head import TgaHead
from pathlib import Path
class DynamicMGAN(nn.Module):
    def __init__(self,hidden_dim=128,num_heads=8,d_att=512):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.d_att=d_att
        print("Creating heads...")
        self.heads=nn.ModuleList([
            TgaHead(hidden_dim)
            for _ in range(num_heads)
        ])
        print("Heads created")
        self.W_o=nn.Linear(hidden_dim*num_heads,d_att)
    
    def forward(self,x,edge_index):
        head_outputs=[]
        attention_maps=[]
        for head in self.heads:
            out,alpha=head(x,edge_index)
            head_outputs.append(out)
            attention_maps.append(alpha)
        
        multi_head=torch.cat(head_outputs,dim=1)

        E_d=self.W_o(multi_head)

        return E_d,attention_maps
    

# %%
model=DynamicMGAN()
x=torch.randn(29,128)
DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/dow30/dynamic_graphs.pt")
graphs=torch.load(DIR)
graph=graphs[0]
E_d,attn=model(x,graph["edge_index"])

print(E_d.shape)

# %%
