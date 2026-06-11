#%%
from pathlib import Path
import numpy as np
import pandas as pd
import torch

EMBED_PATH=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/company_info/")
SAVE_PATH = Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/")

THRESHOLD=0.50

def build_static_graph(embeddings,threshold=0.5):
    corr_matrix=np.corrcoef(embeddings)
    N=corr_matrix.shape[0]
    index=[]
    weight=[]
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            corr=corr_matrix[i,j]
            if corr>threshold:
                index.append([i,j])
                weight.append(corr)

    edge_index=torch.tensor(index,dtype=torch.long).t().contiguous()
    edge_weight=torch.tensor(weight,dtype=torch.float32)
    return edge_index,edge_weight

if __name__=="__main__":
    DATASETS={"dow30":0.392,"ndx100":0.3568,"sse50":0.5039}
    for ds in DATASETS:
        embeddings=np.load(EMBED_PATH/f"{ds}/{ds}_text_embeddings.npy")
        print("Embeddings:",embeddings.shape)
        edge_index, edge_weight = build_static_graph(embeddings,DATASETS.get(ds))
        print("edge_index:", edge_index.shape)
        print("edge_weight:", edge_weight.shape)

        graph = {
        "edge_index": edge_index,
        "edge_weight": edge_weight
        }
        torch.save(graph,SAVE_PATH/f"{ds}/static_graph.pt")
        print(f"{ds} static graph constructed and saved")
        print(f"Saved -> {SAVE_PATH}")

