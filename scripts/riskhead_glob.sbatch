#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=gpunode5
#SBATCH --output=/home/goyalpoonam/logs/%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/%x_%j.err
# Track B (Sharpe-loss + risk-aware head) fine-tune from a pre-trained MSE
# checkpoint. One job = one (MODEL, HORIZON) pair.
#
# Required env vars (passed via sbatch --export):
#   MODEL : DLinear | iTransformer | GCFormer | PatchTST | AdaPatch | TFT | VanillaTransformer
#   H     : 5 | 20 | 60 | 120 | 240
#
# Optional env vars (with defaults):
#   EPOCHS   = 30        (Phase 1 [0..4] MSE warmup, Phase 2 [5..14] alpha=0.3, Phase 3 [15+] alpha=0.7)
#   BATCH    = 512       (256 for transformers if OOM)
#   LR       = 5e-5      (smaller than 1e-4 baseline because we start from a converged backbone)
#   PATIENCE = 8
#
# gpunode5 excluded: instant-fail node (verified on prior runs).

set -e
echo "=== Track B fine-tune | Host: $(hostname) | Partition: $SLURM_JOB_PARTITION | Job: $SLURM_JOB_ID | Started: $(date) ==="

: "${EPOCHS:=30}"
: "${BATCH:=512}"
: "${LR:=5e-5}"
: "${PATIENCE:=8}"

if [[ -z "$MODEL" || -z "$H" ]]; then
    echo "ERROR: MODEL and H must be set via sbatch --export." >&2
    exit 2
fi

INIT_CKPT="checkpoints/${MODEL}_global_H${H}.pth"
echo "MODEL=$MODEL H=$H EPOCHS=$EPOCHS BATCH=$BATCH LR=$LR INIT_CKPT=$INIT_CKPT"

source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate sr_opt

cd ~/SR_optimization

# Sanity: init checkpoint must exist (we run with --init_from). If it does
# not, fail fast — there is no MSE baseline to fine-tune from.
if [[ ! -f "$INIT_CKPT" ]]; then
    echo "ERROR: init checkpoint not found at $INIT_CKPT" >&2
    exit 3
fi

# Train cache may need preprocessing on first job.
python preprocess_global_cache.py 2>&1 | tail -5

stdbuf -oL -eL python -u train.py \
    --model "$MODEL" --method global --horizon "$H" \
    --device auto --batch_size "$BATCH" --epochs "$EPOCHS" \
    --lr "$LR" --lradj type3 --patience "$PATIENCE" --use_amp \
    --use_risk_head --risk_head_lookback 20 --risk_head_d_hidden 64 \
    --init_from "$INIT_CKPT"

echo "=== Track B fine-tune done: $MODEL H${H} | $(date) ==="
