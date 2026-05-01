"""Standalone resume script: load a Round-2 checkpoint and run only Round 3
of sequential training. Use this when a sequential job hits the 12h SLURM
time limit during Round 3 — the trainer saves the best checkpoint at the
end of every round, so we can pick up from there.

This script is INTENTIONALLY separate from train.py — it does not modify
any existing code path. Calls into engine.trainer.Trainer with rounds=1,
after manually loading the round-2 checkpoint.

Usage:
    python scripts/train_resume_r3.py \
        --model TFT --horizon 120 \
        --resume_ckpt checkpoints/TFT_sequential_H120.pth
"""
import os
import sys
import argparse
import torch

# Project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from config import SEQ_LEN, MODEL_SAVE_DIR
from data_loader import UnifiedDataLoader
from models import model_dict
from engine.trainer import Trainer
from train import get_config_for_model
from engine.early_stopping import adjust_learning_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--resume_ckpt", required=True,
                   help="Path to round-2 checkpoint to load before starting R3.")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs_per_stock", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lradj", default="type3")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--device", default="auto")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--save_suffix", default="_r3resume",
                   help="Appended to checkpoint filename to avoid overwriting "
                        "the round-2 checkpoint we just loaded.")
    args = p.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # Build data loader (sequential)
    args.rounds = 1
    args.method = "sequential"
    args.model_name = args.model
    loader_obj = UnifiedDataLoader(seq_len=SEQ_LEN, horizon=args.horizon,
                                   batch_size=args.batch_size)
    val_loader, test_loader = loader_obj.get_val_test_loaders_mmap()

    # Build model + load Round-2 checkpoint
    configs = get_config_for_model(args.model, args.horizon)
    model = model_dict[args.model](configs).to(device)
    print(f"Loading R2 checkpoint: {args.resume_ckpt}", flush=True)
    if not os.path.exists(args.resume_ckpt):
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_ckpt}")
    state = torch.load(args.resume_ckpt, map_location=device)
    model.load_state_dict(state)

    # Save path for the resumed (post-R3) checkpoint — different name so we
    # don't accidentally clobber the R2 we just loaded.
    save_path = os.path.join(
        MODEL_SAVE_DIR,
        f"{args.model}_sequential_H{args.horizon}{args.save_suffix}.pth")
    print(f"Resumed checkpoint will be saved to: {save_path}", flush=True)

    # Build trainer; set its rounds=1 (one round = "Round 3")
    trainer = Trainer(args, model, device)

    # Manually advance the optimizer LR schedule by 2 rounds so the resumed
    # round runs at the LR Round 3 would have used. With lradj=type3 + lr=1e-4,
    # LR stays at 1e-4 for rounds 1/2/3 (no decay until round 4), so this is
    # technically a no-op but we apply it for robustness in case the schedule
    # changes.
    adjust_learning_rate(trainer.optimizer, 1, args)
    adjust_learning_rate(trainer.optimizer, 2, args)

    print("\nStarting Round 3 (resumed) over ~302 stocks "
          f"({args.epochs_per_stock} epochs/stock)", flush=True)
    trainer.train_sequential(loader_obj, val_loader, test_loader, save_path)

    print("=== Resume R3 complete ===", flush=True)


if __name__ == "__main__":
    main()
