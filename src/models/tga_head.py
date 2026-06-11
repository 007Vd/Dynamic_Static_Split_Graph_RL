#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax # type: ignore
# %%
class TgaHead(nn.Module):
    def __init__(self,hidden_dim=128):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.W = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )
        self.a=nn.Parameter(
            torch.empty(2*hidden_dim,1))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu=nn.LeakyReLU(0.2)
    
    def forward(self,x,edge_index):
        h = self.W(x)
        src=edge_index[0]
        dst=edge_index[1]
        h_src=h[src]
        h_dst=h[dst]
        

        edge_features=torch.cat([h_src,h_dst],dim=1)
        scores=(edge_features@self.a).squeeze(-1)

        scores=self.leaky_relu(scores)

        alpha=softmax(scores,dst)
        messages=alpha.unsqueeze(-1)*h_src

        N=x.shape[0]
        out=torch.zeros(N,self.hidden_dim,device=x.device)
        out.index_add_(0,dst,messages)

        return out,alpha
