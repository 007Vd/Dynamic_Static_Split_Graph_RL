import torch
import torch.nn as nn

class DGDRL(nn.Module):
    def __init__(self,lstm_ha,dynamic_mgan,static_mgan,fusion,actor,critic):
        super().__init__()
        self.lstm_ha=lstm_ha
        self.dynamic_mgan=dynamic_mgan
        self.static_mgan=static_mgan
        self.fusion=fusion
        self.actor=actor
        self.critic=critic

    def forward(self,X,dynamic_edge_index,static_edge_index):
        X = X.unsqueeze(0)
        #TEMPORAL ENCODING E_l
        E_l,_=self.lstm_ha(X)
        # print(E_l.shape)
        E_l = E_l.squeeze(0)
        #DYNAMIC BRANCH E_d
        E_d,_=self.dynamic_mgan(E_l,dynamic_edge_index)

        #STATIC BRANCH E_s
        E_s,_=self.static_mgan(E_l,static_edge_index)

        #FUSION E_final
        E_final=self.fusion(E_l,E_d,E_s)

        #ACTOR
        weights,log_prob=self.actor(E_final)
        
        #CRITIC
        value=self.critic(E_final)

        return (weights,log_prob,value,E_final)