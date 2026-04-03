import os
import glob
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SP500WindowDataset(Dataset):
    """
    Loads many CSV files and creates (file, start_index) windows.

    Each __getitem__ returns:
      observed_data: (K, L)
      observed_mask: (K, L)  (all ones here)
      observed_tp:   (L,)    (time positions)
      meta: dict (optional)
    """

    def __init__(
        self,
        root_dir: str = "../../data/sp500_individual_gbm/",
        seq_len: int = 252,
        stride: int = 1,
        columns: Tuple[str, ...] = ("date", "log_adj_close"),
        date_format: str = "%d/%m/%Y",
        time_mode: str = "index_norm",  # "index", "index_norm", "date_ordinal"
        cache_data: bool = False,
        drop_incomplete: bool = True,
    ):
        self.root_dir = root_dir
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.columns = columns
        self.date_format = date_format
        self.time_mode = time_mode
        self.cache_data = cache_data
        self.drop_incomplete = drop_incomplete

        self.files = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No CSV files found in: {root_dir}")

        # Optional cache: file_path -> np.ndarray shape (T, K)
        self._cache: Dict[str, np.ndarray] = {}

        # Build an index of all windows across all files: (file_idx, start)
        self.index: List[Tuple[int, int]] = []
        for fi, fp in enumerate(self.files): #fi = file id, fp = file path
            T = self._get_length(fp)
            if drop_incomplete:
                max_start = T - self.seq_len
                # if the sequence cannot be complete in length is skipped
                if max_start < 0:
                    continue
                for s in range(0, max_start + 1, self.stride): #window: s[s:s+seq_len)
                    self.index.append((fi, s))
            # else:
            #     # Allow last partial window (we'll pad) - not recommended unless needed
            #     for s in range(0, T, self.stride):
            #         self.index.append((fi, s))

        if len(self.index) == 0:
            raise RuntimeError(
                "No windows were created. Likely seq_len is larger than all series lengths."
            )

    def _get_length(self, fp: str) -> int:
        # Fast length check without full parsing: read only Date column
        df = pd.read_csv(fp, usecols=[self.columns[0]])
        return len(df)

    def _load_file(self, fp: str) -> np.ndarray:
        """
        Returns:
          x: (T, K) float32
        """
        if self.cache_data and fp in self._cache:
            return self._cache[fp]

        df = pd.read_csv(fp)

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} is missing columns: {missing}")

        x_cols = list(self.columns[1:])
        x = df[x_cols].to_numpy(dtype=np.float32)  # (T, K)

        return x

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, start = self.index[idx]
        fp = self.files[file_idx]
        x = self._load_file(fp)  # (T, K)

        end = start + self.seq_len
        if end <= len(x):
            x_win  = x[start:end]                                                   # (L, K)
            tp_win = np.linspace(0.0, 1.0, self.seq_len, dtype=np.float32)          # (L,) window-local
            mask   = np.ones_like(x_win, dtype=np.float32)
        # else:
        #     # Padding path (only if drop_incomplete=False)
        #     x_win = x[start:]             # (<=L,K)
        #     tp_win = tp[start:]           # (<=L,)
        #     pad = self.seq_len - len(x_win)
        #     x_win = np.pad(x_win, ((0, pad), (0, 0)), mode="constant") # padding with zeros
        #     tp_win = np.pad(tp_win, (0, pad), mode="edge") # we are repeating the last value
        #     mask = np.zeros_like(x_win, dtype=np.float32)
        #     mask[: self.seq_len - pad, :] = 1.0

        # Convert to torch and transpose to (K,L)
        observed_data = torch.from_numpy(x_win).transpose(0, 1)  # (K,L)
        observed_mask = torch.from_numpy(mask).transpose(0, 1)   # (K,L)
        observed_tp = torch.from_numpy(tp_win)                   # (L,)

        meta = {"file": os.path.basename(fp), "start": start}

        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "observed_tp": observed_tp,
            "meta": meta,
        }


def csdi_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]: # stacking them as to obtain (B,K,L)
    # Stack to (B,K,L) and (B,L)
    observed_data = torch.stack([b["observed_data"] for b in batch], dim=0)
    observed_mask = torch.stack([b["observed_mask"] for b in batch], dim=0)
    observed_tp = torch.stack([b["observed_tp"] for b in batch], dim=0)
    meta = [b["meta"] for b in batch]
    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "observed_tp": observed_tp,
        "meta": meta,
    }


def make_dataloader(
    root_dir: str,
    batch_size: int = 32,
    seq_len: int = 252,
    stride: int = 5,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    ds = SP500WindowDataset(
        root_dir=root_dir,
        seq_len=seq_len,
        stride=stride,
        time_mode="index_norm",   # keep tp stable; Date feature still included in observed_data
        cache_data=False,         # set True if you have enough RAM
        drop_incomplete=True,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=csdi_collate_fn,
        drop_last=True,
    )