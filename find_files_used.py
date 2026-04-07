import torch
import glob, os

# Match your run exactly
root_dir  = "./data/fake_fts_processed"
seq_len   = 64
stride    = 64
seed      = 42                  # from config
subset_size = 16
val_split_ratio = None          # commented out in your config

# 1. Reconstruct the flat index (same logic as SP500WindowDataset.__init__)
files = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
index = []
for fi, fp in enumerate(files):
    import pandas as pd
    T = len(pd.read_csv(fp, usecols=["date"]))
    max_start = T - seq_len
    if max_start < 0:
        continue
    for s in range(0, max_start + 1, stride):
        index.append((fi, s))

dataset_size = len(index)

# 2. Same permutation as the training script
rng = torch.Generator().manual_seed(seed)
all_indices = torch.randperm(dataset_size, generator=rng).tolist()

# 3. Same val carve-out logic
train_pool = all_indices  # val_split_ratio is None

# 4. Same subset selection
train_indices = train_pool[:subset_size]

# 5. Decode back to (file, window_start)
print(f"Total dataset windows: {dataset_size}")
print(f"Training on {subset_size} windows:\n")
for rank, idx in enumerate(train_indices):
    fi, start = index[idx]
    fname = os.path.basename(files[fi])
    print(f"  [{rank:2d}] dataset_idx={idx:5d}  file={fname}  window_start={start}")
