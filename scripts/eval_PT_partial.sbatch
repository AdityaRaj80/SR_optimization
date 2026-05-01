#!/bin/bash
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_PT_partial
#SBATCH --output=/home/goyalpoonam/logs/%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/%x_%j.err

set -e
echo "=== Started: $(date) ==="
source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate sr_opt
cd ~/SR_optimization

# Sequential — only H=5/60/120 done; H=20 and H=240 still training
python evaluate_extended.py --model PatchTST --method sequential --horizons 5 60 120
cp results/extended_metrics.csv results/extended_metrics_PatchTST_sequential.csv
cp results/per_stock_metrics.csv results/per_stock_metrics_PatchTST_sequential.csv

# Global — all 5 horizons done
python evaluate_extended.py --model PatchTST --method global --horizons 5 20 60 120 240
cp results/extended_metrics.csv results/extended_metrics_PatchTST_global.csv
cp results/per_stock_metrics.csv results/per_stock_metrics_PatchTST_global.csv

# Financial metrics (Sharpe, MDD, etc.) — partial seq, full global
python evaluate_financial.py --model PatchTST --methods sequential --horizons 5 60 120
python evaluate_financial.py --model PatchTST --methods global --horizons 5 20 60 120 240

echo "=== Done: $(date) ==="
