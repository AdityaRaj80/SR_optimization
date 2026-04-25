import argparse
import os
import torch
import pandas as pd
from models import model_dict
from config import *
from data_loader import UnifiedDataLoader
from engine.trainer import Trainer
from engine.evaluator import evaluate

def get_config_for_model(model_name, horizon):
    # Base config from global config
    base = {
        "pred_len": horizon,
        "context_len": SEQ_LEN,
        "enc_in": len(FEATURES)
    }
    
    if model_name == "PatchTST":
        base.update(PATCHTST_CONFIG)
    elif model_name == "TFT":
        base.update(TFT_CONFIG)
    elif model_name == "AdaPatch":
        base.update(ADAPATCH_CONFIG)
    elif model_name == "GCFormer":
        base.update(GCFORMER_CONFIG)
    elif model_name == "iTransformer":
        base.update(ITRANSFORMER_CONFIG)
    elif model_name == "VanillaTransformer":
        base.update(VANILLA_TRANSFORMER_CONFIG)
    elif model_name == "TimesNet":
        base.update(TIMESNET_CONFIG)
    elif model_name == "DLinear":
        base.update(DLINEAR_CONFIG)
        
    return base

def main():
    parser = argparse.ArgumentParser(description='CIKM Benchmarking Framework')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--method', type=str, default='global', choices=['global', 'sequential'], help='Training method')
    parser.add_argument('--horizon', type=int, required=True, help='Prediction horizon')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda:X, hpc, or auto)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs')
    parser.add_argument('--rounds', type=int, default=3, help='Rounds for sequential method')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lradj', type=str, default='type3', help='LR adjust strategy')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--adapatch_alpha', type=float, default=0.5, help='Alpha param for AdaPatch loss')
    parser.add_argument('--epochs_per_stock', type=int, default=10, help='Epochs per stock per round in sequential training')
    parser.add_argument('--max_stocks', type=int, default=None, help='Limit number of training stocks (for timing tests)')
    args = parser.parse_args()

    if args.device == 'hpc':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere/Hopper (H100)
            torch.set_float32_matmul_precision('high')
    elif args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    args.model_name = args.model # Save model name

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("Loading datasets...")
    loader = UnifiedDataLoader(seq_len=SEQ_LEN, horizon=args.horizon, batch_size=args.batch_size,
                               max_stocks=args.max_stocks)
    val_loader, test_loader = loader.get_val_test_loaders()
    
    if val_loader is None or test_loader is None:
        print("Failed to initialize val/test loaders. Exiting.")
        return

    # Initialize model
    print(f"Initializing {args.model} for horizon {args.horizon}...")
    configs = get_config_for_model(args.model, args.horizon)
    model_class = model_dict[args.model]
    model = model_class(configs).to(device)

    # Setup trainer
    trainer = Trainer(args, model, device)
    
    save_name = f"{args.model}_{args.method}_H{args.horizon}.pth"
    save_path = os.path.join(MODEL_SAVE_DIR, save_name)

    # Train
    print(f"Starting {args.method} training...")
    if args.method == 'global':
        train_loader = loader.get_global_train_loader()
        model = trainer.train_global(train_loader, val_loader, test_loader, save_path)
    elif args.method == 'sequential':
        stock_train_loaders = loader.get_sequential_train_loaders()
        model = trainer.train_sequential(stock_train_loaders, val_loader, test_loader, save_path)
        
    print("Training complete. Evaluating on test set...")
    model.load_state_dict(torch.load(save_path))
    test_metrics = evaluate(model, test_loader, device)
    
    # Save results
    res_path = os.path.join(RESULTS_DIR, f"{args.method}_results.csv")
    res_df = pd.DataFrame([{
        "Model": args.model,
        "Horizon": args.horizon,
        "MSE": test_metrics['mse'],
        "MAE": test_metrics['mae'],
        "R2": test_metrics['r2']
    }])
    
    if os.path.exists(res_path):
        res_df.to_csv(res_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(res_path, index=False)
        
    print(f"Final Test Metrics for {args.model} (H={args.horizon}): MSE={test_metrics['mse']:.5f}, MAE={test_metrics['mae']:.5f}, R2={test_metrics['r2']:.5f}")

if __name__ == "__main__":
    main()
