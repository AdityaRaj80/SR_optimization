#!/bin/bash
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_AP
#SBATCH --output=/home/goyalpoonam/logs/%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/%x_%j.err

set -e
echo "=== Started: $(date) ==="
source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate sr_opt
cd ~/SR_optimization

python evaluate_extended.py --model AdaPatch --method sequential --horizons 5 20 60 120 240
cp results/extended_metrics.csv results/extended_metrics_AdaPatch_sequential.csv
cp results/per_stock_metrics.csv results/per_stock_metrics_AdaPatch_sequential.csv

python evaluate_extended.py --model AdaPatch --method global --horizons 5 20 60 120 240
cp results/extended_metrics.csv results/extended_metrics_AdaPatch_global.csv
cp results/per_stock_metrics.csv results/per_stock_metrics_AdaPatch_global.csv

python evaluate_financial.py --model AdaPatch --methods sequential global --horizons 5 20 60 120 240

echo "=== Done: $(date) ==="
