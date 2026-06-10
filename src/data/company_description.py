#%%
import yfinance as yf
import pandas as pd
from pathlib import Path


dow30=["AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX",
    "DIS","GS","HD","HON","IBM","JNJ","JPM","KO",
    "MCD","MMM","MRK","MSFT","NKE","NVDA","PG","SHW",
    "TRV","UNH","V","VZ","WMT"]

sse50 = [
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
ndx100 = [
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

datasets = {
    "dow30": dow30,
    "sse50": sse50,
    "ndx100": ndx100
}
for dataset_name, tickers in datasets.items():
    rows = []
    PATH_DIR=Path(f"/Users/007vd/Downloads/DAU/dgdrl_paper/data/company_info/{dataset_name}")
    PATH_DIR.mkdir(parents=True,exist_ok=True)

    for ticker in tickers:
        

        try:
            stock = yf.Ticker(ticker)

            info = stock.info

            rows.append({
            "ticker": ticker,
            "description":
                info.get("longBusinessSummary","")
            })

            print(f"{ticker} done")

        except Exception as e:
            print(ticker,e)
  
        df = pd.DataFrame(rows)

    df.to_csv(
    PATH_DIR/f"{dataset_name}_descriptions.csv",
    index=False)

    print(f"====={dataset_name} DONE====")

        


# %%
