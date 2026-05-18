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
        time_mode: str = "global_index",  # "index_norm", "global_index"
        cache_data: bool = True, # set True if you have enough RAM to speed up loading (stores file data in self._cache), otherwise we trigger a reload and re-read of csv
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

        # Optional cache: file_path -> (x: np.ndarray (T,K), dates: np.ndarray (T,) str)
        self._cache: Dict[str, Any] = {}

        # Single pass: collect all dates and file lengths together
        all_dates: set = set()
        file_lengths: List[int] = []
        for fp in self.files:
            df = pd.read_csv(fp, usecols=[self.columns[0]])
            all_dates.update(df[self.columns[0]].tolist())
            file_lengths.append(len(df))
        self._file_lengths: List[int] = file_lengths
        self.date_to_idx: Dict[str, int] = {
            d: i for i, d in enumerate(
                sorted(all_dates, key=lambda s: pd.to_datetime(s, format='%d/%m/%Y'))
            )
        }

        # Build an index of all windows across all files: (file_idx, start)
        self.index: List[Tuple[int, int]] = []
        for fi, fp in enumerate(self.files): #fi = file id, fp = file path
            T = file_lengths[fi]
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

    def _load_file(self, fp: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          x:     (T, K) float32
          dates: (T,)   str — date strings in the order they appear in the file
        """
        if self.cache_data and fp in self._cache:
            return self._cache[fp]

        df = pd.read_csv(fp)

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} is missing columns: {missing}")

        x_cols = list(self.columns[1:])
        x = df[x_cols].to_numpy(dtype=np.float32)  # (T, K)
        dates = df[self.columns[0]].to_numpy(dtype=str)  # (T,)

        if self.cache_data:
            self._cache[fp] = (x, dates)

        return x, dates

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, start = self.index[idx]
        fp = self.files[file_idx]
        x, dates = self._load_file(fp)  # (T, K), (T,)

        end = start + self.seq_len
        if end <= len(x):
            x_win    = x[start:end]                                                 # (L, K)
            dates_win = dates[start:end]                                            # (L,)
            tp_win   = np.array([self.date_to_idx[d] for d in dates_win],
                                dtype=np.float32)                                   # (L,) global ints
            mask     = np.ones_like(x_win, dtype=np.float32)
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

    def split_indices(
        self,
        val_fraction: float,
        seed: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """
        Returns (train_indices, val_indices) as index lists into self.index.

        For each file a single contiguous block of timesteps is held out as the
        validation zone.  The block is placed at a uniformly random interior
        position (reproducible via `seed`) such that at least one complete
        training window fits to the left of the block.

        Classification of a window (fi, start):
          - entirely left  of block  [start+seq_len <= val_start]  → training
          - entirely right of block  [start >= val_end]            → training
          - entirely inside block    [start >= val_start and
                                      start+seq_len <= val_end]    → validation
          - straddles either boundary                              → discarded
            (prevents any single data point appearing in both splits)

        Files where T < seq_len + max(seq_len, floor(T*val_fraction)) are too
        short to yield both window types; all their windows go to training.
        """
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

        rng = np.random.default_rng(seed)

        file_val_start: Dict[int, int] = {}
        file_val_end:   Dict[int, int] = {}
        n_skipped = 0

        for fi, T in enumerate(self._file_lengths):
            val_block_size = max(self.seq_len, int(T * val_fraction))
            # val_start must satisfy:
            #   val_start >= seq_len        (room for ≥1 training window on the left)
            #   val_start + val_block_size <= T  (block fits in the file)
            lo = self.seq_len
            hi = T - val_block_size
            if lo > hi:
                file_val_start[fi] = -1   # sentinel: training-only file
                n_skipped += 1
                continue
            val_start = int(rng.integers(lo, hi + 1))
            file_val_start[fi] = val_start
            file_val_end[fi]   = val_start + val_block_size

        if n_skipped:
            import warnings
            warnings.warn(
                f"split_indices: {n_skipped}/{len(self._file_lengths)} files are too short "
                f"to yield both training and validation windows; "
                f"all their windows go to training.",
                stacklevel=2,
            )

        train_indices: List[int] = []
        val_indices:   List[int] = []

        for i, (fi, start) in enumerate(self.index):
            vs = file_val_start[fi]
            if vs == -1:
                train_indices.append(i)
                continue
            ve  = file_val_end[fi]
            end = start + self.seq_len

            if end <= vs:
                train_indices.append(i)        # entirely left of val block
            elif start >= ve:
                train_indices.append(i)        # entirely right of val block
            elif start >= vs and end <= ve:
                val_indices.append(i)          # entirely inside val block
            # else: straddles a boundary — discard to prevent leakage

        return train_indices, val_indices


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
    columns: Tuple[str, ...] = ("date", "log_adj_close"),
) -> DataLoader:
    ds = SP500WindowDataset(
        root_dir=root_dir,
        seq_len=seq_len,
        stride=stride,
        columns=columns,
        time_mode="global_index",  # global calendar position shared across all tickers
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