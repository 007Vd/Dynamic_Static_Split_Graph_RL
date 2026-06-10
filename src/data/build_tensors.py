

from pathlib import Path
import numpy as np
import pandas as pd


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


def build_dataset(dataset_name: str):

    DATA_DIR = Path(
        f"/Users/007vd/Downloads/DAU/dgdrl_paper/data/processed/{dataset_name}"
    )

    SAVE_DIR = Path(
        f"/Users/007vd/Downloads/DAU/dgdrl_paper/data/tensors/{dataset_name}"
    )

    SAVE_DIR.mkdir(parents=True,exist_ok=True)

    files = sorted(DATA_DIR.glob("*.csv"))

    stocks = []

    stock_features = []
    stock_returns = []

    for file in files:

        stock = file.stem
        stocks.append(stock)

        df = pd.read_csv(file)

        feature_array = df[FEATURES].values.astype(np.float32)

        stock_features.append(feature_array)

        close = df["Close"].values.astype(np.float32)

        returns = (close[1:] / close[:-1]) - 1

        stock_returns.append(returns)

    X = np.stack(
        stock_features,
        axis=1
    )

    future_returns = np.stack(
        stock_returns,
        axis=1
    )

    np.save(
        SAVE_DIR / "X.npy",
        X
    )

    np.save(
        SAVE_DIR / "future_returns.npy",
        future_returns
    )

    pd.Series(stocks).to_csv(
        SAVE_DIR / "stocks.csv",
        index=False
    )

    print(f"\n===== {dataset_name.upper()} =====")
    print("X shape:", X.shape)
    print("Returns shape:", future_returns.shape)

    print("Saved:")
    print(SAVE_DIR / "X.npy")
    print(SAVE_DIR / "future_returns.npy")


if __name__ == "__main__":

    DATASETS = [
        "dow30",
        "ndx100",
        "sse50"
    ]

    for dataset in DATASETS:
        build_dataset(dataset)