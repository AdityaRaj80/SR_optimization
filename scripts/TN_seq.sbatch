#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/goyalpoonam/logs/%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/%x_%j.err

set -e
echo === Host: $(hostname) | Partition: $SLURM_JOB_PARTITION | Job: $SLURM_JOB_ID | Started: $(date) ===

source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate sr_opt

cd ~/SR_optimization
python preprocess_global_cache.py --valtest 2>&1 | tail -5

# TimesNet-specific: batch_size=128 to stay within 80GB headroom under
# Inception_Block_V1 + FFT-derived period reshapes. Sequential H=5 OOMed at 512.
python -u train.py --model $MODEL --method $METHOD --horizon $H     --device auto --batch_size 128 --rounds 3 --epochs_per_stock 20     --epochs 60 --lr 1e-4 --lradj type3 --patience 10 --use_amp

echo === Job done: $(date) ===
