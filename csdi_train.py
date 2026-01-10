#!/usr/bin/env python3
"""
Converted from CSDI_Test.ipynb into an executable training script (no plots).

Assumed project layout (same as notebook):
- ../configs/base_csdi.yaml
- ../data/{ticker}_{start}_{end}_processed.csv
- ../../CSDI/  (cloned CSDI repo with main_model.py)
- ../checkpoints/ (for checkpoints + history json)
"""

from pathlib import Path
import argparse
import sys
import os
import json
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim


# -------------------------
# Config helpers
# -------------------------
def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found at: {config_path}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML: {e}") from e


# -------------------------
# Path setup (matches notebook approach)
# -------------------------
def setup_import_paths() -> None:
    # 1) go back to parent folder
    p1 = Path.cwd()
    # 2) go back to parent of the parent
    p2 = Path.cwd().parent
    # 3) enter folder "CSDI" (assumed to be inside that parent-of-parent)
    csdi_dir = p2 / "CSDI"

    if not csdi_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {csdi_dir}")

    # Add CSDI to Python import path so imports work from anywhere
    sys.path.insert(0, str(csdi_dir))

    dataset_dir = p1 / "data"
    sys.path.insert(0, str(dataset_dir))


# -------------------------
# Main training
# -------------------------
def main():
    # ---- Experiment config (from notebook) ----
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--start-date", type=str, default="2010-01-01")
    p.add_argument("--end-date", type=str, default="2024-12-31")
    p.add_argument("--sequence-length", type=int, default=252)
    p.add_argument("--validation-ratio", type=float, default=0.2)
    p.add_argument("--execute-train", type=bool, default=True)

    args = p.parse_args()

    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date


    sequence_length = args.sequence_length
    validation_ratio = args.validation_ratio

    execute_train = args.execute_train
    # checkpoint_path = "../checkpoints/checkpoint_epoch_40.pth"  # used only if execute_train=False
    # num_samples = 100  # used only if execute_train=False

    config_path = "./configs/base_csdi.yaml"
    data_path = f"./data/{ticker}_{start_date}_{end_date}_processed.csv"
    checkpoints_dir = Path("./checkpoints")
    outputs_dir = Path("./outputs")

    # checkpoints_dir.mkdir(parents=True, exist_ok=True)
    # outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Imports that depend on sys.path setup ----
    setup_import_paths()
    from main_model import CSDI_base  # noqa: E402
    from yahoo_data import FinancialDataset  # noqa: E402

    # ---- Load config ----
    config = load_config(config_path)
    target_dim = config["model"]["target_dim"]
    beta_start = config["diffusion"]["beta_start"]
    print(f"Initialized with target_dim: {target_dim}, beta_start: {beta_start}")

    # ---- Model wrapper (from notebook) ----
    class CSDI_Financial(CSDI_base):
        def __init__(self, config, device, target_dim=5):
            super().__init__(target_dim, config, device)

        def process_data(self, batch):
            observed_data = batch["observed_data"].to(self.device).float()
            observed_mask = batch["observed_mask"].to(self.device).float()
            observed_tp = batch["timepoints"].to(self.device).float()
            gt_mask = batch["gt_mask"].to(self.device).float()

            # Optional conditioning mask (if dataset provides it)
            cond_mask = batch.get("cond_mask", observed_mask).to(self.device).float()

            cut_length = torch.zeros(len(observed_data)).long().to(self.device)

            return (observed_data, observed_mask, observed_tp, gt_mask, cond_mask, cut_length)

    # ---- Device ----
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data ----
    df = pd.read_csv(data_path)
    print(f"Loaded data: {data_path}")
    print(df.head())
    print("Descriptive Statistics:")
    print(df.describe())

    # ---- Dataloaders (chronological split) ----
    def create_dataloaders(df, seq_len, val_rat, batch_size):
        # Some versions of FinancialDataset accept close_idx/forecast_horizon; some do not.
        try:
            full_dataset = FinancialDataset(dataset=df, seq_len=seq_len, close_idx=3, forecast_horizon=None)
        except TypeError:
            full_dataset = FinancialDataset(dataset=df, seq_len=seq_len)

        total_len = len(full_dataset)
        val_len = int(total_len * val_rat)
        train_len = total_len - val_len

        train_dataset = Subset(full_dataset, range(0, train_len))
        val_dataset = Subset(full_dataset, range(train_len, total_len))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    batch_size = config["train"]["batch_size"]
    train_loader, val_loader = create_dataloaders(df, sequence_length, validation_ratio, batch_size=batch_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Init model/optimizer ----
    model = CSDI_Financial(config, device, target_dim=target_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])

    # ---- Train ----
    if execute_train:
        today_str = datetime.now().strftime("%Y-%m-%d")
        history = {"train_loss": [], "val_loss": [], "volatility_error": []}

        epochs = int(config["train"]["epochs"])
        itr_per_epoch = int(config["train"].get("itr_per_epoch", 10**18))

        for epoch in range(epochs):
            model.train()
            cumulative_loss = 0.0
            steps = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch}", total=min(len(train_loader), itr_per_epoch))
            for i, batch in enumerate(progress):
                optimizer.zero_grad(set_to_none=True)

                loss = model(batch, is_train=1)
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                cumulative_loss += loss_val
                steps += 1
                progress.set_postfix(loss=f"{loss_val:.4f}")

                if steps >= itr_per_epoch:
                    break

            avg_loss = cumulative_loss / max(steps, 1)
            history["train_loss"].append(avg_loss)

            # Validation (single batch, like notebook style)
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                vloss = model(val_batch, is_train=1)
                val_loss = float(vloss.item()) if hasattr(vloss, "item") else float(vloss)
                history["val_loss"].append(val_loss)

                # Quick volatility diagnostic (matches notebook; global std)
                samples, observed, target_mask, _, _ = model.evaluate(val_batch, n_samples=5)
                gen_vol = float(samples.std().item())
                real_vol = float(observed.std().item())
                vol_err = abs(gen_vol - real_vol)
                history["volatility_error"].append(vol_err)

            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Vol Error: {vol_err:.4f}")

            # Save checkpoints periodically
            if epoch % 10 == 0:
                ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch}_{today_str}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # Save final checkpoint
        final_ckpt = checkpoints_dir / f"checkpoint_epoch_{epochs-1}_{today_str}.pth"
        torch.save(model.state_dict(), final_ckpt)
        print(f"Saved final checkpoint: {final_ckpt}")

        # Save training history
        hist_path = checkpoints_dir / f"{ticker}_training_history_{today_str}.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved history: {hist_path}")

    # else:
    #     # Optional: evaluation-only mode (kept from notebook, no plots)
    #     print("Training skipped as per configuration. Loading checkpoint for sampling.")
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #     model.eval()

    #     test_batch = next(iter(val_loader))
    #     with torch.no_grad():
    #         samples, observed, target_mask, obs_mask, obs_tp = model.evaluate(test_batch, n_samples=num_samples)

    #     print(f"samples shape: {tuple(samples.shape)}")

    #     save_path = outputs_dir / f"{ticker}_{start_date}_{end_date}_csdi_out.pt"
    #     torch.save(
    #         {
    #             "samples": samples.detach().cpu(),
    #             "observed": observed.detach().cpu(),
    #             "target_mask": target_mask.detach().cpu(),
    #             "obs_mask": obs_mask.detach().cpu(),
    #             "obs_tp": obs_tp.detach().cpu(),
    #             "n_samples": num_samples,
    #         },
    #         save_path,
    #     )
    #     print(f"Saved outputs: {save_path}")


if __name__ == "__main__":
    main()
