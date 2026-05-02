#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=gpunode5
#SBATCH --output=/home/goyalpoonam/logs/%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/%x_%j.err
# gpunode5 excluded: every job dispatched to it instant-fails with exit 0:53
# (no logs, ~1s elapsed). Confirmed by 8/8 failure pattern vs 9/9 success on
# gpunode6. Suspected hardware/driver fault on gpunode5; remove this line
# once admin confirms the node is fixed.

set -e
echo "=== Host: $(hostname) | Partition: $SLURM_JOB_PARTITION | Job: $SLURM_JOB_ID | Started: $(date) ==="

source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate sr_opt

cd ~/SR_optimization
python preprocess_global_cache.py --valtest 2>&1 | tail -5

stdbuf -oL -eL python -u train.py --model $MODEL --method $METHOD --horizon $H \
    --device auto --batch_size 512 --rounds 3 --epochs_per_stock 20 \
    --epochs 60 --lr 1e-4 --lradj type3 --patience 10 --use_amp

echo "=== Job done: $(date) ==="
