#%%
import yfinance as yf
from pathlib import Path
import pandas as pd

RAW_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/raw/sse50")
RAW_DIR.mkdir(parents=True,exist_ok=True)


SSE50 = [
    "600000.SS","600009.SS","600016.SS","600028.SS","600030.SS",
    "600031.SS","600036.SS","600048.SS","600050.SS","600104.SS",
    "600111.SS","600196.SS","600276.SS","600309.SS","600438.SS",
    "600519.SS","600547.SS","600570.SS","600585.SS","600588.SS",
    "600690.SS","600703.SS","600745.SS","600809.SS",
    "600887.SS","601012.SS","601066.SS","601088.SS","601138.SS",
    "601166.SS","601169.SS","601186.SS","601211.SS","601229.SS",
    "601288.SS","601318.SS","601328.SS","601336.SS","601398.SS",
    "601601.SS","601628.SS","601668.SS","601688.SS","601818.SS",
    "601857.SS","601888.SS","601899.SS","601919.SS","603259.SS"
]

START = "2020-01-01"
END   = "2023-12-31"

#%%
for ticker in SSE50:
    print(f"downloading {ticker} data")

    df= yf.download(ticker,
                    start=START,
                    end=END,
                    auto_adjust=False,
                 progress=False
                    )
    
    if(len(df)==0):
        print(f"{ticker} Failed")
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    columns=["Date",'Close', 'High', 'Low', 'Open', 'Volume']
    df=df[columns]
    print(f"Saving {ticker} -> {RAW_DIR/f'{ticker}.csv'}")
    df.to_csv(RAW_DIR/f"{ticker}.csv") 

files=list(Path(RAW_DIR).glob("*.csv"))

#%%
print(len(SSE50))
df=pd.read_csv(RAW_DIR/"600000.SS.csv")
print(df.head())
print(df.columns)
# %%
for ticker in SSE50:

    df = pd.read_csv(RAW_DIR/f"{ticker}.csv")
    
    print(ticker, len(df))
# %%
