#%%
import yfinance as yf
from pathlib import Path
import pandas as pd

RAW_DIR=Path("/Users/007vd/Downloads/DAU/dgdrl_paper/data/raw/ndx100")
RAW_DIR.mkdir(parents=True,exist_ok=True)


NDX100 = [
    "AAPL","ADBE","ADI","ADP","ADSK","AEP","AMAT",
    "AMD","AMGN","AMZN","ASML","AVGO","AZN","BIIB",
    "BKNG","CDNS","CMCSA","COST","CPRT","CRWD","CSCO",
    "CSX","CTAS","CTSH","DDOG","DXCM","EA","EXC",
    "FANG","FAST","FTNT","GILD","GOOG","GOOGL",
    "HON","IDXX","INTC","INTU","ISRG","KDP","KHC","KLAC",
    "LIN","LRCX","LULU","MAR","MCHP","MDB","MDLZ","MELI",
    "META","MNST","MRVL","MSFT","MU","NFLX","NVDA","ODFL",
    "ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL",
    "QCOM","REGN","ROP","ROST","SBUX","SNPS","TEAM","TMUS",
    "TSLA","TTD","TXN","VRTX","WBD","WDAY","XEL","ZS"
]

START = "2020-01-01"
END   = "2023-12-31"

#%%
for ticker in NDX100:
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
print(len(NDX100))
df=pd.read_csv(RAW_DIR/"AAPL.csv")
print(df.head())
print(df.columns)
# %%
for ticker in NDX100:

    df = pd.read_csv(RAW_DIR/f"{ticker}.csv")
    
    print(ticker, len(df))
# %%
