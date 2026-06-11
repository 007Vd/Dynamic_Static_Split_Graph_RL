#%%
import torch
import torch.nn as nn

class RMLP(nn.Module):
    def __init__(self,input_dim=1024,d_att=512,num_layers=4,dropout=0.1):
        super().__init__()
        hidden_dim=d_att//num_layers

        self.layers= nn.ModuleList()
        self.layers.append(
            nn.Linear(input_dim,hidden_dim))
        
        for _ in range(num_layers-1):
            self.layers.append(
                nn.Linear(hidden_dim,hidden_dim)
            )
        
        self.dropout=nn.Dropout(dropout)
        self.activation=nn.ReLU()
        self.score_head=nn.Linear(hidden_dim*num_layers,1)

    def forward(self,x):
        h=x
        outputs=[]
        for layer in self.layers:
            h=layer(h)
            h=self.activation(h)
            h=self.dropout(h)
            outputs.append(h)

        concat_features=torch.cat(outputs,dim=-1)
        score=self.score_head(concat_features)

        return score




