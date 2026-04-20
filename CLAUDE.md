# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis: generative modelling of S&P 500 log-returns for those companies with 40 years or more of historical data. This is achieved by using a CSDI-style diffusion model with a GBM-inspired forward VE SDE. The model uses a 1D-UNet architecture with Transformer as in CSDI, continuous-time noise schedules (VE/VP/GBM), and is trained on sliding windows of log-adjusted-returns of log-prices in an unconditional manner.

## Common Commands

### Training
```bash
python csdi_train.py --config configs/replication.yaml
# override individual params:
python csdi_train_modified.py --config configs/replication.yaml --epochs 50 --batch_size 64 --lr 1e-4
```

### Sample Generation
```bash
python generate_samples.py \
    --checkpoint_folder replication \
    --checkpoint_name <name>.pt \
    --n_samples 250 \
    --seed 42
# output lands in data/generated/<checkpoint_subfolder>/
```

### HPC Job Submission (SLURM)
```bash
sbatch csdi_train.sh        # training (20h, 64GB, 32 CPUs, dsba partition)
sbatch generate_samples.sh  # generation (4h, 4 GPUs, stud partition)
```

### Filtering Data Windows
```bash
python src/utils/filter_windows.py  # creates stratified subsets by variance/kurtosis/period
```

## Architecture

### Pipeline
```
CSV data (log_adj_close)
  → SP500WindowDataset   (sliding windows, time embeddings)
  → CSDIModel            (side-info: time + feature embeddings)
  → diff_CSDI            (1D U-Net: Conv → ResidualBlock → TransformerEncoder → Conv)
  → Diffusion_Processes  (forward/reverse SDE steps, ODE sampler)
  → MSE loss on score with importance-sampling weighting
```

### Key Source Files
| File | Role |
|------|------|
| `csdi_train_modified.py` | Main training script (preferred over `csdi_train.py` for debugging) |
| `generate_samples.py` | Reverse-diffusion sample generation |
| `src/models/diff_models.py` | `diff_CSDI` — 1D U-Net backbone |
| `src/models/model_core.py` | `CSDIModel` — conditioning and side-info construction |
| `src/utils/WIP_processes.py` | `Diffusion_Processes` — forward/reverse SDE orchestration |
| `src/utils/WIP_SDE.py` | `VESDE`, `VPSDE`, `SubVPSDE`, `GBMLogSDE` implementations |
| `src/utils/dataloader.py` | `SP500WindowDataset` — sliding-window dataset with caching |
| `src/utils/sde_utils.py` | Importance-sampling weight calculation |
| `configs/replication.yaml` | Default hyperparameter config |

### Config Structure (`configs/replication.yaml`)
Nested YAML with sections: `data`, `model`, `diffusion` (network arch), `process` (SDE type + schedule), `train`, `wandb`. Key params: `seq_len=2048`, `stride=400`, `sigma_min/max`, SDE type (`ve`/`vp`/`gbm`). Some of these are specific for parameter replication.

### SDE Types
- `ve` — variance-exploding (exponential sigma schedule)
- `vp` — variance-preserving
- 'subvp' - sub-variance-preserving
- 'gbm' - reduced to VE SDE

### Training Infrastructure
- **Optimizer:** AdamW with AMP (`torch.cuda.amp`)
- **EMA** tracking of model weights
- **W&B** logging (project `csdi-gbm`, team `thesis-giacomo-negri`)
- **Checkpoints** saved to `checkpoints/<run_name>/`

## Data Layout
- `data/replication/` — raw S&P 500 CSV files (columns: `date`, `log_adj_close`)
- `data/filtered_windows/` — stratified subsets (high/low/moderate variance, kurtosis, pre/post-2001)
- `data/generated/<checkpoint>/` — synthetic samples from generation runs
- `data/fake_gbm/` — toy GBM paths for sanity checks

## Notebooks
Located in `notebooks/`. Used for EDA, training-vs-generated distribution analysis, and diagnostics. Not part of any automated pipeline.

## Dependencies
No root `requirements.txt`; dependencies are tracked per W&B run. Key packages: `torch`, `lightning`, `wandb`, `optuna`, `linear_attention_transformer`, `numpy`, `pandas`, `matplotlib` (forced to `Agg` backend).
