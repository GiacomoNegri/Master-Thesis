#!/bin/bash
#SBATCH --job-name=train_gbm
#SBATCH --account=3155287
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --nodelist=gnode04
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3155287@studbocconi.it

set -x
mkdir -p out err

echo "Job running on:"
hostname
whoami

module load miniconda3
source /software/miniconda3/etc/profile.d/conda.sh
conda activate thesis

echo "Running on:"
hostname
nvidia-smi || True

echo "Python executable:"
which python
python --version

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

cd /home/3155287/Thesis/Master-Thesis


echo "PWD:"
pwd

srun python -u csdi_train_modified.py --config configs/csdi_gbm.yaml --epochs 5 --data_root ./data/fake_individual_gbm --train_subset_ratio 0.1
echo "Python exit code: $?"

echo "Job finished."

