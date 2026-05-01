#!/bin/bash
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=TFT_seq_H120_R3
#SBATCH --output=/home/goyalpoonam/logs/%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/%x_%j.err

set -e
echo "=== Resume R3 started: $(date) | Host: $(hostname) ==="

source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate sr_opt

cd ~/SR_optimization
python preprocess_global_cache.py --valtest 2>&1 | tail -5

# Resume from R2 checkpoint and run only Round 3
stdbuf -oL -eL python -u scripts/train_resume_r3.py \
    --model TFT --horizon 120 \
    --resume_ckpt checkpoints/TFT_sequential_H120.pth \
    --batch_size 512 --epochs_per_stock 20 \
    --lr 1e-4 --lradj type3 --patience 10 --use_amp \
    --device auto

echo "=== Done: $(date) ==="
