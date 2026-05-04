#!/bin/bash
# Stage Track B (Sharpe-loss + risk-aware head) fine-tune campaign.
#
# Usage:
#   bash scripts/submit_riskhead_campaign.sh GCFormer
#   bash scripts/submit_riskhead_campaign.sh GCFormer iTransformer
#   bash scripts/submit_riskhead_campaign.sh ALL
#
# For each (MODEL, HORIZON) pair this submits one job running
# scripts/riskhead_glob.sbatch with --export MODEL,H. Partition is alternated
# H100 / H200 to fan jobs across both QOS pools (each pool allows 2 concurrent).
#
# Total job count per model = 5 (H=5,20,60,120,240).

set -e
cd ~/SR_optimization

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <MODEL>... | ALL"
    exit 1
fi

if [[ "$1" == "ALL" ]]; then
    MODELS=(DLinear iTransformer GCFormer PatchTST AdaPatch TFT VanillaTransformer)
else
    MODELS=("$@")
fi

HORIZONS=(5 20 60 120 240)

# Alternate partitions to spread load across H100 + H200 QOS pools.
PARTITIONS=(gpu_h100_4 gpu_h200_8 gpu_h100_4 gpu_h200_8 gpu_h100_4)

JOB_NUM=0
for MODEL in "${MODELS[@]}"; do
    INIT_OK=true
    # Pre-flight: ensure all init checkpoints exist on disk before queuing.
    for H in "${HORIZONS[@]}"; do
        CKPT="checkpoints/${MODEL}_global_H${H}.pth"
        if [[ ! -f "$CKPT" ]]; then
            echo "[SKIP $MODEL H$H] missing init checkpoint at $CKPT"
            INIT_OK=false
        fi
    done
    if [[ "$INIT_OK" == "false" ]]; then
        echo "[ERROR] $MODEL has missing init checkpoints. Sync with scp first."
        echo "        Skipping all $MODEL horizons."
        continue
    fi

    for i in "${!HORIZONS[@]}"; do
        H="${HORIZONS[$i]}"
        PART="${PARTITIONS[$i]}"
        JOB_NAME="${MODEL}_RH_H${H}"
        echo "[submit] $JOB_NAME on $PART (init from checkpoints/${MODEL}_global_H${H}.pth)"
        sbatch --partition="$PART" \
               --job-name="$JOB_NAME" \
               --export=ALL,MODEL="$MODEL",H="$H" \
               scripts/riskhead_glob.sbatch
        JOB_NUM=$((JOB_NUM + 1))
        sleep 1
    done
done

echo
echo "Submitted $JOB_NUM job(s). Monitor with:"
echo "  squeue -u \$USER --format='%.10i %.30j %.10P %.10M %.8T %.10R'"
