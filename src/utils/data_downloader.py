import os
import re
import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUTPUT_DIR_RAW = os.path.join(os.path.dirname(__file__), "../../data/raw_replication/")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/log_replication/")
MIN_YEARS = 40

os.makedirs(OUTPUT_DIR_RAW, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    table = pd.read_html(pd.io.common.StringIO(response.text))
    df = table[0]
    print(f"Table of companies:\n",df)

    tickers = df["Symbol"].astype(str).tolist()
    # Keep only clean tickers (no dots, dashes, slashes, etc.)
    tickers = [t for t in tickers if re.fullmatch(r"[A-Z0-9]+", t)]
    return tickers


def main():
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} clean tickers.")
    print("Tickers list: ", tickers)

    for ticker in tickers:
        try:
            df = yf.download(
                tickers=ticker,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                print(f"Skipping {ticker}: no data returned.")
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel("Ticker")

            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]

            # Require at least MIN_YEARS of history
            last_date = df.index[-1]
            cutoff = last_date - pd.DateOffset(years=MIN_YEARS)

            if df.index[0] > cutoff:
                span_years = (df.index[-1] - df.index[0]).days / 365.25
                print(f"Skipping {ticker}: only {span_years:.1f} years of data.")
                continue
            
            # Saving raw dataset
            raw_result = pd.DataFrame({
                "date": df.index,
                "adj_close": df["Close"]
            })

            # commented out to asses if missing values where present
            close = df["Close"]#.dropna()

            # Log-transform the adjusted close price
            log_close = np.log(close)

            result = pd.DataFrame({
                "date": log_close.index,
                "log_adj_close": log_close.values,
            })

            first_day = log_close.index[0].strftime("%Y-%m-%d")
            last_day = log_close.index[-1].strftime("%Y-%m-%d")

            file_path = os.path.join(OUTPUT_DIR_RAW, f"raw_{ticker}_{first_day}_{last_day}.csv")
            raw_result.to_csv(file_path, index=False)

            print(f"Saved raw {ticker}: {len(raw_result)} rows ({first_day} to {last_day}).")
    
            file_path = os.path.join(OUTPUT_DIR, f"log_{ticker}_{first_day}_{last_day}.csv")
            result.to_csv(file_path, index=False)

            print(f"Saved {ticker}: {len(result)} rows ({first_day} to {last_day}).")

        except Exception as e:
            print(f"Failed to process {ticker}: {e}")


if __name__ == "__main__":
    main()