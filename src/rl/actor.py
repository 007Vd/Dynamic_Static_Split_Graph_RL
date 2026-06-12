import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from src.rl.rmlp import RMLP


class Actor(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        d_att=512,
        num_layers=4
    ):
        super().__init__()

        self.rmlp = RMLP(
            input_dim=input_dim,
            d_att=d_att,
            num_layers=num_layers
        )

    def forward(self, E_final):

        # [N,1]
        scores = self.rmlp(E_final)

        # [N]
        scores = scores.squeeze(-1)

        # Dirichlet concentration parameters
        alpha = F.softplus(scores) + 0.01

        dist = Dirichlet(alpha)

        # portfolio weights
        weights = dist.rsample()

        # PPO log probability
        log_prob = dist.log_prob(weights)

        return weights, log_prob