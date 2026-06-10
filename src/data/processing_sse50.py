from pathlib import Path
import pandas as pd

RAW_DIR = Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/raw/sse50")
all_dates = None

for file in RAW_DIR.glob("*.csv"):
    df = pd.read_csv(file)
    dates = pd.to_datetime(df["Date"])
    if all_dates is None:
        all_dates = set(dates)
    else:
        all_dates = all_dates.intersection(set(dates))

print(len(all_dates))
calendar = sorted(list(all_dates))
calendar = pd.DatetimeIndex(calendar)
print(calendar[0])
print(calendar[-1])
print(len(calendar))

ALIGNED_DIR = Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/processed/sse50")

ALIGNED_DIR.mkdir(parents=True,exist_ok=True)

for file in RAW_DIR.glob("*.csv"):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = (df.set_index("Date").reindex(calendar))
    df.index.name = "Date"
    df.to_csv(ALIGNED_DIR/file.name)


for file in ALIGNED_DIR.glob("*.csv"):

    df = pd.read_csv(file)

    print(
        file.stem,
        df.isna().sum().sum()
    )