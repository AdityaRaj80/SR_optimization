#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/goyalpoonam/logs/%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/%x_%j.err

set -e
echo "=== Host: $(hostname) | Partition: $SLURM_JOB_PARTITION | Job: $SLURM_JOB_ID | alpha=$ALPHA | Started: $(date) ==="

source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate sr_opt

cd ~/SR_optimization
python preprocess_global_cache.py --valtest 2>&1 | tail -5

# AdaPatch alpha-sweep variant. Same hyperparams as iT_seq_partition.sbatch
# but with --adapatch_alpha overridable via $ALPHA env var.
stdbuf -oL -eL python -u train.py --model AdaPatch --method $METHOD --horizon $H \
    --device auto --batch_size 512 --rounds 3 --epochs_per_stock 20 \
    --epochs 60 --lr 1e-4 --lradj type3 --patience 10 --use_amp \
    --adapatch_alpha $ALPHA

echo "=== Job done: $(date) ==="
