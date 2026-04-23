import argparse
import os
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description='Run All Models Script')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()

    models = ["PatchTST", "TFT", "AdaPatch", "GCFormer", "iTransformer", "VanillaTransformer", "TimesNet", "DLinear"]
    horizons = [3, 10, 40, 120, 240]
    methods = ["global", "sequential"]
    
    for method in methods:
        for model in models:
            for horizon in horizons:
                cmd = [
                    "python", "train.py",
                    "--model", model,
                    "--method", method,
                    "--horizon", str(horizon),
                    "--device", args.device
                ]
                if args.use_amp:
                    cmd.append("--use_amp")
                
                print(f"\n{'='*50}\nRunning: {' '.join(cmd)}\n{'='*50}")
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error executing: {cmd}")
                    print(e)
                time.sleep(2) # brief pause between runs

if __name__ == "__main__":
    main()
