#%%
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from torch_geometric.data import Data


def compute_return(close_prices):
    returns=(close_prices[1:]-close_prices[:-1])-1

    return returns

def build_corr_matrix(return_window):
    corr,_=spearmanr(return_window,axis=0)
    n_stocks=return_window.shape[1]
    corr=corr[:n_stocks,:n_stocks]

    return corr


def corr_to_graph(corr_matrix,threshold=0.6):
    edge_index=[]
    edge_weight=[]
    n=corr_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            corr=corr_matrix[i,j]
            if np.isnan(corr):
                continue
            if abs(corr)>=threshold:
                edge_index.append([i,j])
                edge_weight.append(corr)

    edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
    edge_weight=torch.tensor(edge_weight,dtype=torch.float)

    return edge_index,edge_weight


def build_dynamic_graph(X,graph_window=40,threshold=0.6):
    CLOSE_IDX=3
    close_prices=X[:,:,3]
    returns=compute_return(close_prices)
    graphs=[]
    for t in range(graph_window,len(returns)):
        return_window=returns[t-graph_window:t]
        corr_matrix=build_corr_matrix(return_window)

        edge_index,edge_weight=corr_to_graph(corr_matrix,threshold)
        graphs.append({
            "edge_index":edge_index,
            "edge_weight":edge_weight
        })
    return graphs


# %%
if __name__=="__main__":
    # X=np.load("/Users/007vd/Downloads/DAU/dgdrl_paper/data/tensors/dow30/X.npy")
    # graphs=build_dynamic_graph(X,40,0.6)
    # print(graphs[0]["edge_index"].shape)
    # print(graphs[0]["edge_weight"].shape)

    DATASET=["dow30","ndx100","sse50"]
   
    
    DGRAPH_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs")

    for ds in DATASET:
        X_PATH=Path(f"/Users/007vd/Downloads/DAU/dgdrl_paper/data/tensors/{ds}/X.npy")
        X=np.load(X_PATH)
        graphs=build_dynamic_graph(X,40,0.6)
        print(f"{ds} graphs made successfully")

        SAVE_DIR=Path(DGRAPH_DIR/f"{ds}")
        print(f"saving {ds} dynamic graphs to {str(SAVE_DIR)}/")
        SAVE_DIR.mkdir(parents=True,exist_ok=True)
        torch.save(graphs,SAVE_DIR/f"dynamic_graphs.pt")

# %%
print("For DOW30")
graphs = torch.load(
    "/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/dow30/dynamic_graphs.pt"
)

print(len(graphs))

g = graphs[0]

print(g["edge_index"].shape)
print(g["edge_weight"].shape)

print(g["edge_weight"].min())
print(g["edge_weight"].max())

#%%
print("For SSE50")
graphs = torch.load(
    "/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/sse50/dynamic_graphs.pt"
)

print(len(graphs))

g = graphs[0]

print(g["edge_index"].shape)
print(g["edge_weight"].shape)

print(g["edge_weight"].min())
print(g["edge_weight"].max())

#%%
print("For NDX100")
graphs = torch.load(
    "/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/ndx100/dynamic_graphs.pt"
)

print(len(graphs))

g = graphs[0]

print(g["edge_index"].shape)
print(g["edge_weight"].shape)

print(g["edge_weight"].min())
print(g["edge_weight"].max())