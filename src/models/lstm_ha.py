import torch
import torch.nn as nn
import torch.nn.functional as F

class HistoricalAttention(nn.Module):
    def __init__(self,hidden_dim:int):
        super().__init__()
        self.attn=nn.Linear(hidden_dim,1)

    def forward(self,h):
        scores=self.attn(h)
        weights=F.softmax(scores,dim=1)
        context=torch.sum(weights*h,dim=1)
        return context,weights

class LSTMHA(nn.Module):
    def __init__(self,input_dim=5,hidden_dim=128,num_layers=1):
        super().__init__()
        self.hidden_dim=hidden_dim

        self.lstm=nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.ha=HistoricalAttention(hidden_dim)
    
    def forward(self,x):
        B,N,L,F=x.shape
        x=x.reshape(B*N,L,F)
        h,_=self.lstm(x)
        E_l,attn=self.ha(h)
        E_l=E_l.reshape(B,N,self.hidden_dim)
        attn=attn.reshape(B,N,L,1)

        return E_l,attn
