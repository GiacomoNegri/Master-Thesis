import os
import pandas as pd

src_dir = os.path.join(os.path.dirname(__file__), "../../data/fake_individual_gbm")
dst_dir = os.path.join(os.path.dirname(__file__), "../../data/fake_individual_gbm_close")
os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(src_dir, fname), usecols=["Date", "Close"])
    df = df.rename(columns={"Date": "date", "Close": "log_adj_close"})

    df.to_csv(os.path.join(dst_dir, fname), index=False)

print(f"Done. Processed {len(os.listdir(dst_dir))} files into '{dst_dir}'.")
