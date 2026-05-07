# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis: conditional generative modelling of S&P 500 OHLC log-returns using a CSDI-style diffusion model trained under the EDM objective (Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models"). The model uses a transformer-based architecture (CSDI backbone), EDM preconditioning and loss weighting, and is trained on sliding windows of OHLC log-returns in `predict_close` mode: Open, High, Low are observed conditioning features; Close is the target to generate.

## Current Phase

**EDM training and generation** (`edm` branch). The SDE-based training path (VE/VP/GBM) and associated files are legacy — present in the repository but no longer part of the active pipeline.

## Common Commands

### Training
```bash
python csdi_train.py --config configs/ohlc_conditional.yaml
# override individual params:
python csdi_train.py --config configs/ohlc_conditional.yaml --epochs 50 --batch_size 64 --lr 1e-4
```

### Sample Generation
```bash
python generate_samples.py \
    --checkpoint_folder replication \
    --checkpoint_name <name>.pt \
    --n_samples 250 \
    --seed 42
# output lands in data/generated/<checkpoint_subfolder>/
# uses the EDM sampler only; SDE-based generation is legacy
```

### HPC Job Submission (SLURM)
```bash
sbatch csdi_train.sh        # training (20h, 64GB, 32 CPUs, dsba partition)
sbatch generate_samples.sh  # generation (4h, 4 GPUs, stud partition)
```

### Dataset generation and pre-processing
```bash
python src/utils/replication_data_downloader.py  # collect the dataset from the S&P 500 companies
python src/utils/ohlc_to_returns.py              # compute log-returns for Close and transform Open, High, Low
```

## Architecture

### Pipeline
```
CSV data (date, close, open, high, low — log-returns)
  → SP500WindowDataset   (sliding windows, global calendar time embeddings)
  → CSDIModel            (EDM preconditioning; side-info: time + feature embeddings + cond_mask)
  → diff_CSDI            (backbone: Conv → ResidualBlock → TransformerEncoder → Conv)
  → EDMLoss              (log-normal sigma sampling, lambda(sigma)-weighted MSE on Close only)
```

### Key Source Files
| File | Role |
|------|------|
| `csdi_train.py` | Main training script |
| `generate_samples.py` | EDM-sampler-based sample generation |
| `src/models/diff_models.py` | `diff_CSDI` — transformer backbone |
| `src/models/model_core.py` | `CSDIModel` — EDM preconditioning, side-info, conditional input formatting |
| `src/training/edm_loss.py` | `EDMLoss` — Karras et al. 2022 training objective |
| `src/utils/dataloader.py` | `SP500WindowDataset` — sliding-window dataset |
| `configs/ohlc_conditional.yaml` | Active hyperparameter config (OHLC, predict_close mode) |

### Config Structure (`configs/ohlc_conditional.yaml`)
Nested YAML with sections: `data`, `model`, `diffusion` (network arch), `process` (kept for sampler compatibility), `train`, `edm` (P_mean, P_std), `wandb`. Key params: `seq_len=512`, `stride=100`, `sigma_data=1.0`, `mask_mode=predict_close`, `close_idx=0`.

### EDM Preconditioning
At each training step one sigma per sample is drawn from `ln σ ~ N(P_mean, P_std²)`. The model receives `c_in`-scaled noisy input and outputs the MMSE denoised estimate `D_x = c_skip·x_t + c_out·F_x`. Loss is the lambda(sigma)-weighted MSE restricted to the Close channel via `target_mask`.

### Masking Modes
- `predict_close` — condition on Open/High/Low, always predict Close (`close_idx` from `config["data"]["close_idx"]`)
- `random` — randomly hide `cond_min_ratio`–`cond_max_ratio` of all entries as targets
- `unconditional` — no conditioning; model must generate the full joint distribution

### Training Infrastructure
- **Optimizer:** AdamW with optional AMP (`torch.cuda.amp`)
- **LR schedule:** optional CosineAnnealingLR
- **Early stopping:** optional, controlled by `early_stop_patience`
- **W&B** logging (project `csdi-ohlc`, team `thesis-giacomo-negri`)
- **Checkpoints** saved to `checkpoints/ohlc_conditional/`

## Data Layout
- `data/fake_fts_processed/` — preprocessed OHLC log-return CSV files (columns: `date`, `close`, `open`, `high`, `low`); active training data
- `data/replication/` — raw S&P 500 CSV files (columns: `date`, `log_adj_close`); legacy
- `data/generated/<checkpoint>/` — synthetic samples from generation runs
- `data/filtered_windows/` — stratified subsets (high/low/moderate variance, kurtosis, pre/post-2001)
- `data/fake_gbm/` — toy GBM paths for sanity checks

## Notebooks
Located in `notebooks/`. Used for EDA, training-vs-generated distribution analysis, and diagnostics. Not part of any automated pipeline.

## Dependencies
No root `requirements.txt`; dependencies are tracked per W&B run. Key packages: `torch`, `wandb`, `linear_attention_transformer`, `numpy`, `pandas`, `matplotlib` (forced to `Agg` backend).
