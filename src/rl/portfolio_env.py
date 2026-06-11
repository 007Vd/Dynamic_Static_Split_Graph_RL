#%%
import torch

class PortfolioEnv:
    def __init__(self):
        pass

    def step(self,weights,future_returns):
        reward=torch.sum(weights*future_returns)

        return reward
    

