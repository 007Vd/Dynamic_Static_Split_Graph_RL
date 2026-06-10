#%%
from sentence_transformers import SentenceTransformer 
import pandas as pd
import numpy as np
dataset=["dow30","sse50","ndx100"]

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

for ds in dataset:
    df=pd.read_csv(f"/Users/007vd/Downloads/DAU/dgdrl_paper/data/company_info/{ds}/{ds}_descriptions.csv")

    embeddings=model.encode(df["description"].tolist(),convert_to_numpy=True)
    np.save(f"/Users/007vd/Downloads/DAU/dgdrl_paper/data/company_info/{ds}/{ds}_text_embeddings.npy",embeddings)
    print(f"{ds} has been completed ans saved")
# %%
