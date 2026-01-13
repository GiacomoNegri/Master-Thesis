#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import json
from datetime import datetime

import yaml
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim


# -------------------------
# Config helpers
# -------------------------
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Path setup (ONLY for your dataset module, not CSDI repo)
# -------------------------
def setup_import_paths() -> None:
    # your notebook added ./data to import yahoo_data.py; keep that
    p1 = Path.cwd()
    dataset_dir = p1 / "data"
    sys.path.insert(0, str(dataset_dir))
    sys.path.insert(0, str(p1 / "src" / "models"))


# -------------------------
# Batch adapter: FinancialDataset -> csdi_scratch expected batch
# -------------------------
def to_scratch_batch(batch: dict, seq_len: int) -> dict:
    x = batch["observed_data"]       # (B, K, L)
    obs_mask = batch["observed_mask"]# (B, K, L)
    cond = batch["cond_mask"]        # (B, K, L)
    tp = batch["timepoints"]         # (L,) typically, from your dataset

    # Convert to (B, L, K)
    x = x.permute(0, 2, 1).contiguous()
    obs_mask = obs_mask.permute(0, 2, 1).contiguous()
    cond = cond.permute(0, 2, 1).contiguous()

    # timepoints: (L,) -> (B, L)
    if tp.dim() == 1:
        tp = tp.unsqueeze(0).expand(x.shape[0], -1).contiguous()
    elif tp.dim() != 2:
        raise ValueError(f"Unexpected timepoints shape: {tp.shape}")

    return {
        "observed_data": x.float(),
        "observed_mask": obs_mask.float(),
        "cond_mask": cond.float(),
        "timepoints": tp.float(),
    }

@torch.no_grad()
def masked_rmse_from_scratch(model, batch_scratch, n_samples: int = 8) -> float:
    """
    Evaluate imputation quality on target positions = observed_mask - cond_mask.
    """
    samples = model.impute(batch_scratch, n_samples=n_samples)  # (B, S, L, K)
    pred = samples.mean(dim=1)  # (B, L, K)

    x0 = batch_scratch["observed_data"]
    obs_mask = batch_scratch["observed_mask"]
    cond_mask = (batch_scratch["cond_mask"] * obs_mask)
    target_mask = (obs_mask - cond_mask).clamp(0.0, 1.0)

    err = (pred - x0) * target_mask
    denom = target_mask.sum().clamp(min=1.0)
    rmse = torch.sqrt((err ** 2).sum() / denom).item()
    return float(rmse)

@torch.no_grad()
def evaluate(model, loader, device, seq_len: int, eval_samples: int, max_batches: int | None = None):
    model.eval()
    loss_sum = 0.0
    rmse_sum = 0.0
    n = 0

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        batch_s = to_scratch_batch(batch, seq_len=seq_len)
        batch_s = {k: v.to(device) for k, v in batch_s.items()}

        vloss = model(batch_s)
        loss_sum += float(vloss.item())

        rmse_sum += masked_rmse_from_scratch(model, batch_s, n_samples=eval_samples)
        n += 1

    return loss_sum / max(n, 1), rmse_sum / max(n, 1)


def main():
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

    config_path = "./configs/base_csdi.yaml"
    data_path = f"./data/{ticker}_{start_date}_{end_date}_processed.csv"
    checkpoints_dir = Path("./checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ---- Imports depending on sys.path ----
    setup_import_paths()
    from yahoo_data import FinancialDataset, create_dataloader # noqa: E402

    # ---- Import your scratch model (no CSDI repo needed) ----
    from src.models.csdi_scratch import CSDIFromScratch, DiffusionConfig  # noqa: E402

    # ---- Load config ----
    config = load_config(config_path)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data ----
    df = pd.read_csv(data_path)
    print(f"Loaded data: {data_path}")
    print(df.head())

    batch_size = int(config["train"]["batch_size"])
    train_loader, val_loader = create_dataloader(df = df, seq_len = sequence_length, close_idx = 3, val_rat = validation_ratio, batch_size=batch_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Infer K from first batch AFTER adapting to scratch format ----
    first_batch = next(iter(train_loader))
    first_batch_s = to_scratch_batch(first_batch, seq_len=sequence_length)
    K = first_batch_s["observed_data"].shape[-1]
    print(f"Inferred feature dim K = {K}")

    # ---- Init scratch model ----
    # Map yaml fields if you want; otherwise use safe defaults for feasibility.
    cfg = DiffusionConfig(
        num_steps=int(config.get("diffusion", {}).get("num_steps", 50)),
        beta_start=float(config.get("diffusion", {}).get("beta_start", 1e-4)),
        beta_end=float(config.get("diffusion", {}).get("beta_end", 2e-2)),
        d_model=int(config.get("model", {}).get("d_model", 128)),
        nhead=int(config.get("model", {}).get("nhead", 4)),
        num_layers=int(config.get("model", {}).get("num_layers", 4)),
        dim_feedforward=int(config.get("model", {}).get("dim_feedforward", 256)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    )

    model = CSDIFromScratch(K=K, cfg=cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]), weight_decay=1e-4)
    
    itr_per_epoch = len(train_loader)
    print("Batches used per epoch:", min(len(train_loader), itr_per_epoch), "/", len(train_loader))

    # ---- Train ----
    if execute_train:
        today_str = datetime.now().strftime("%Y-%m-%d")
        history = {"train_loss": [], "val_loss": [], "val_masked_rmse": []}

        epochs = int(config["train"]["epochs"])
        itr_per_epoch = int(config["train"].get("itr_per_epoch", 10**18))
        eval_samples = int(config["train"].get("eval_samples", 8))

        step_global = 0

        for epoch in range(epochs):
            model.train()
            cumulative_loss = 0.0
            steps = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch}", total=min(len(train_loader), itr_per_epoch))
            for i, batch in enumerate(progress):
                batch_s = to_scratch_batch(batch, seq_len=sequence_length)
                batch_s = {k: v.to(device) for k, v in batch_s.items()}

                optimizer.zero_grad(set_to_none=True)
                loss = model(batch_s)  # <-- scratch API
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_val = float(loss.item())
                cumulative_loss += loss_val
                steps += 1
                step_global += 1
                progress.set_postfix(loss=f"{loss_val:.4f}")

                if steps >= itr_per_epoch:
                    break

            avg_loss = cumulative_loss / max(steps, 1)
            history["train_loss"].append(avg_loss)

            # Validation (single batch)
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                val_s = to_scratch_batch(val_batch, seq_len=sequence_length)
                val_s = {k: v.to(device) for k, v in val_s.items()}

                vloss = model(val_s)
                val_loss = float(vloss.item())
                history["val_loss"].append(val_loss)

                rmse = masked_rmse_from_scratch(model, val_s, n_samples=eval_samples)
                history["val_masked_rmse"].append(rmse)
            
            val_loss, rmse = evaluate(
                model, val_loader, device,
                seq_len=sequence_length,
                eval_samples=eval_samples,
                max_batches=int(config["train"].get("val_eval_batches", 10**18))
            )
            history["val_loss"].append(val_loss)
            history["val_masked_rmse"].append(rmse)

            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val masked RMSE: {rmse:.4f}")

            # Save checkpoint periodically
            if epoch % 10 == 0:
                ckpt_path = checkpoints_dir / f"scratch_ckpt_epoch_{epoch}_{today_str}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "cfg": cfg.__dict__,
                        "K": K,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

        # Save final checkpoint
        final_ckpt = checkpoints_dir / f"scratch_ckpt_epoch_{epochs-1}_{today_str}.pt"
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epochs - 1, "cfg": cfg.__dict__, "K": K},
            final_ckpt,
        )
        print(f"Saved final checkpoint: {final_ckpt}")

        # Save training history
        hist_path = checkpoints_dir / f"{ticker}_scratch_history_{today_str}.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved history: {hist_path}")


if __name__ == "__main__":
    main()
