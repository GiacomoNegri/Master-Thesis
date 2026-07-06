"""Plot Open/High/Low/Close for one ticker across the three data pipeline stages:
raw replication, processed (log-returns), and normalized.

Usage:
    python plot_ohlc_pipeline_stages.py --ticker AAPL
    python plot_ohlc_pipeline_stages.py --ticker AAPL --output-dir images/final/presentation
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STAGES = [
    ("data/SNP500_individual_replication", "raw", "Raw prices"),
    ("data/SNP500_individual_processed_replication", "processed", "Log-returns"),
    ("data/SNP500_individual_normalized_replication", "normalized", "Normalized log-returns"),
]

SEQ_LEN = 200

# Fixed categorical color assignment, consistent across all three plots.
# Close is drawn last (front-most z-order) and rendered with a heavier line
# so it always reads on top of Open/High/Low where they overlap.
COLUMN_STYLE = {
    "open":  {"color": "#2a78d6", "zorder": 2, "linewidth": 0.7, "linestyle": "-"},
    "high":  {"color": "#dc790e", "zorder": 2, "linewidth": 0.7, "linestyle": "--"},
    "low":   {"color": "#0DD11D", "zorder": 2, "linewidth": 0.7, "linestyle": "--"},
    "close": {"color": "#e51818", "zorder": 3, "linewidth": 1.0, "linestyle": "-"},
}
COLUMN_ORDER = ["open", "high", "low", "close"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def find_ticker_csv(folder: str, ticker: str) -> str:
    matches = sorted(glob.glob(os.path.join(folder, f"{ticker}_*.csv")))
    if not matches:
        raise FileNotFoundError(f"No CSV found for ticker '{ticker}' in {folder}")
    return matches[0]


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df.sort_values("date").tail(SEQ_LEN)


def plot_stage(df: pd.DataFrame, ticker: str, stage_label: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for col in COLUMN_ORDER:
        style = COLUMN_STYLE[col]
        ax.plot(
            df["date"], df[col],
            label=col.capitalize(),
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            zorder=style["zorder"],
            solid_capstyle="round",
            dash_capstyle="round",
        )

    ax.set_title(f"{ticker} — {stage_label}", fontsize=13, color=INK_PRIMARY, pad=12)
    ax.set_xlabel("Date", fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("Value", fontsize=11, color=INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.grid(False)

    legend = ax.legend(fontsize=10, framealpha=0.9, facecolor=SURFACE, edgecolor=GRIDLINE)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (base filename prefix)")
    parser.add_argument("--output-dir", default="plots/ohlc_stages", help="Directory to save the three PNGs")
    args = parser.parse_args()

    for folder, tag, stage_label in STAGES:
        csv_path = find_ticker_csv(folder, args.ticker)
        df = load_ohlc(csv_path)
        out_path = os.path.join(args.output_dir, f"{args.ticker}_{tag}.png")
        plot_stage(df, args.ticker, stage_label, out_path)


if __name__ == "__main__":
    main()
