from pathlib import Path
import numpy as np

def create_windows(X:np.ndarray,future_returns:np.ndarray,lookback:int=20):
    T,N,F=X.shape
    X_window=[]
    y_returns=[]
    l_w=lookback
    for t in range(l_w,T-1):
        window=X[t-l_w+1:t+1]
        window=np.transpose(window,(1,0,2))

        X_window.append(window)
        y_returns.append(future_returns[t])

    X_window=np.array(X_window,dtype=np.float32)
    y_returns=np.array(y_returns,dtype=np.float32)

    return X_window,y_returns

if __name__=="__main__":
    DATASET=["dow30","ndx100","sse50"]
    LOOKBACK=20
    for ds in DATASET:
        base=Path(f"/Users/007vd/Downloads/DAU/dgdrl_paper/data/tensors/{ds}")

        X=np.load(base/"X.npy")
        future_returns=np.load(base/"future_returns.npy")
    
        X_windows,y_returns=create_windows(X,future_returns,lookback=LOOKBACK)
        print("\nShapes")
        print("X:", X.shape)
        print("Returns:", future_returns.shape)
    
        print("\nWindowed Shapes")
        print("X_windows:", X_windows.shape)
        print("y_returns:", y_returns.shape)
    
        np.save(base/f"X_windows_{LOOKBACK}.npy",X_windows)
        np.save(base/f"returns_windows_{LOOKBACK}.npy",y_returns)
    
        print("SAVED")
    

