import datafeed
import pandas as pd


tickers = ["DASH", "ETH", "IOT", "LTC", "XRP", "XMR"]


def load_portfolio():
    i = 0
    for t in tickers:
        d = datafeed.getdaydata(t, "BTC")
        if i == 0:
            df = pd.DataFrame(d["close"])
        else:
            df = df.join( d["close"])
    return df



print load_portfolio()
