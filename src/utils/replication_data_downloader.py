import os
import re
import pandas as pd
import requests
import yfinance as yf

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/SNP500_individual/")
MIN_TRADING_DAYS = 1000

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
    tickers = df["Symbol"].astype(str).tolist()
    # Keep only clean tickers (no dots, dashes, slashes, etc.)
    tickers = [t for t in tickers if re.fullmatch(r"[A-Z0-9]+", t)]
    return tickers


def main():
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} clean tickers.")

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

            # Require at least MIN_TRADING_DAYS of history
            cols = ["Open", "High", "Low", "Close"]
            available = [c for c in cols if c in df.columns]
            df = df[available].dropna()

            if len(df) < MIN_TRADING_DAYS:
                print(f"Skipping {ticker}: only {len(df)} trading days.")
                continue

            # yfinance auto_adjust=True already returns adjusted prices;
            # expose them as both "Close" and "Adj Close" for clarity
            result = df[available].copy()
            result.columns = [c.lower().replace(" ", "_") for c in available]
            result.insert(0, "date", df.index.strftime("%d/%m/%Y"))


            first_day = df.index[0].strftime("%Y-%m-%d")
            last_day = df.index[-1].strftime("%Y-%m-%d")
            file_path = os.path.join(OUTPUT_DIR, f"{ticker}_{first_day}_{last_day}.csv")
            result.to_csv(file_path, index=False)
            print(f"Saved {ticker}: {len(result)} rows ({first_day} to {last_day}).")

        except Exception as e:
            print(f"Failed to process {ticker}: {e}")


if __name__ == "__main__":
    main()