from pathlib import Path
import torch
import numpy as np
def align_lstm_and_graphs(X_window,y_returns,graphs,lookback=20,graph_window=40):
    start_index=graph_window-lookback+1
    X_window=X_window[start_index:]
    y_returns=y_returns[start_index:]

    n=min(len(X_window),len(graphs))

    X_window=X_window[:n]
    y_returns=y_returns[:n]
    graphs=graphs[:n]

    return (
        X_window,y_returns,graphs
    )

if __name__=="__main__":

    #DOW30
    DIR_PATH=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/tensors")
    X_windows=np.load(DIR_PATH/"dow30"/"X_windows_20.npy")
    y_returns=np.load(DIR_PATH/"dow30"/"returns_windows_20.npy")
    graphs=torch.load("/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/dow30/dynamic_graphs.pt")
    aligned_X_windows,aligned_y_Returns,aligned_graphs= align_lstm_and_graphs(
        X_windows,y_returns,graphs,lookback=20,graph_window=40)
    
    SAVE_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/aligned/dow30")
    torch.save(
    torch.tensor(aligned_X_windows,dtype=torch.float32),
    SAVE_DIR/"X_aligned.pt"
    )

    torch.save(
    torch.tensor(aligned_y_Returns,dtype=torch.float32),
    SAVE_DIR/"y_aligned.pt"
    )

    torch.save(
    aligned_graphs,
    SAVE_DIR/"dynamic_graphs_aligned.pt"
    )

    #SSE50
    DIR_PATH=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/tensors")
    X_windows=np.load(DIR_PATH/"sse50"/"X_windows_20.npy")
    y_returns=np.load(DIR_PATH/"sse50"/"returns_windows_20.npy")
    graphs=torch.load("/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/sse50/dynamic_graphs.pt")
    aligned_X_windows,aligned_y_Returns,aligned_graphs= align_lstm_and_graphs(
        X_windows,y_returns,graphs,lookback=20,graph_window=40)
    
    SAVE_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/aligned/sse50")
    torch.save(
    torch.tensor(aligned_X_windows,dtype=torch.float32),
    SAVE_DIR/"X_aligned.pt"
    )

    torch.save(
    torch.tensor(aligned_y_Returns,dtype=torch.float32),
    SAVE_DIR/"y_aligned.pt"
    )

    torch.save(
    aligned_graphs,
    SAVE_DIR/"dynamic_graphs_aligned.pt"
    )

    #NDX100
    DIR_PATH=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/tensors")
    X_windows=np.load(DIR_PATH/"ndx100"/"X_windows_20.npy")
    y_returns=np.load(DIR_PATH/"ndx100"/"returns_windows_20.npy")
    graphs=torch.load("/Users/007vd/Downloads/DAU/dgdrl_paper/data/graphs/ndx100/dynamic_graphs.pt")
    aligned_X_windows,aligned_y_Returns,aligned_graphs= align_lstm_and_graphs(
        X_windows,y_returns,graphs,lookback=20,graph_window=40)
    
    SAVE_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/aligned/ndx100")
    torch.save(
    torch.tensor(aligned_X_windows,dtype=torch.float32),
    SAVE_DIR/"X_aligned.pt"
    )

    torch.save(
    torch.tensor(aligned_y_Returns,dtype=torch.float32),
    SAVE_DIR/"y_aligned.pt"
    )

    torch.save(
    aligned_graphs,
    SAVE_DIR/"dynamic_graphs_aligned.pt"
    )
    



  