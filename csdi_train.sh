#!/bin/bash
#SBATCH --job-name=csdi_train
#SBATCH --account=3155287
#SBATCH --partition=dsba
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --out=out/%x_%j.out
#SBATCH --err=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3155287@studbocconi.it

source activate ts_diffusion
cd ~/Thesis/Master-Thesis
srun python -u csdi_train.py --ticker="AAPL" --start-date="2010-01-01" --end-date="2024-12-31"
