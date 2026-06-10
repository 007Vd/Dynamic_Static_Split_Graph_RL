#%%
import yfinance as yf
from pathlib import Path
import pandas as pd

RAW_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/raw")
RAW_DIR.mkdir(parents=True,exist_ok=True)

DOW_30=["AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX",
    "DIS","GS","HD","HON","IBM","JNJ","JPM","KO",
    "MCD","MMM","MRK","MSFT","NKE","NVDA","PG","SHW",
    "TRV","UNH","V","VZ","WMT"]

START = "2020-01-01"
END   = "2023-12-31"

# %%
for ticker in DOW_30:
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

# %%
files=list(Path(RAW_DIR).glob("*.csv"))
len(files)
# %%
df=pd.read_csv(RAW_DIR/"AAPL.csv")
print(df.head())
print(df.columns)
# %%
