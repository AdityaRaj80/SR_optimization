import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW

from engine.early_stopping import EarlyStopping, adjust_learning_rate
from engine.evaluator import evaluate

class Trainer:
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = []
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if self.args.use_amp:
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                with torch.autocast(device_type=device_type):
                    outputs = self.model(batch_x, None)
                    if isinstance(outputs, tuple):
                        if self.args.model_name == 'AdaPatch':
                            pred, orig, dec = outputs
                            loss_pred = self.criterion(pred, batch_y)
                            loss_rec = self.criterion(dec, orig)
                            loss = self.args.adapatch_alpha * loss_pred + (1 - self.args.adapatch_alpha) * loss_rec
                        else:
                            outputs = outputs[0]
                            loss = self.criterion(outputs, batch_y)
                    else:
                        loss = self.criterion(outputs, batch_y)
                
                loss.backward()  # Should use GradScaler for full AMP, but simplified here
                self.optimizer.step()
            else:
                outputs = self.model(batch_x, None)
                if isinstance(outputs, tuple):
                    if self.args.model_name == 'AdaPatch':
                        pred, orig, dec = outputs
                        loss_pred = self.criterion(pred, batch_y)
                        loss_rec = self.criterion(dec, orig)
                        loss = self.args.adapatch_alpha * loss_pred + (1 - self.args.adapatch_alpha) * loss_rec
                    else:
                        outputs = outputs[0]
                        loss = self.criterion(outputs, batch_y)
                else:
                    loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            train_loss.append(loss.item())

        return sum(train_loss) / len(train_loss)

    def train_global(self, train_loader, val_loader, test_loader, save_path):
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.epochs):
            t1 = time.time()
            train_loss = self.train_epoch(train_loader)
            
            val_metrics = evaluate(self.model, val_loader, self.device, self.criterion)
            val_loss = val_metrics["loss"]
            
            test_metrics = evaluate(self.model, test_loader, self.device, self.criterion)
            test_loss = test_metrics["loss"]

            print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} Vali Loss: {val_loss:.5f} Test Loss: {test_loss:.5f} | Time: {time.time()-t1:.2f}s")
            
            early_stopping(val_loss, self.model, save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            
        self.model.load_state_dict(torch.load(save_path))
        return self.model

    def train_sequential(self, stock_train_loaders, val_loader, test_loader, save_path):
        # No early stopping to mimic catastrophic forgetting effect if present
        best_val = float('inf')
        epochs_per_stock = getattr(self.args, 'epochs_per_stock', 10)

        for r in range(self.args.rounds):
            t1 = time.time()
            print(f"Starting Round {r+1}/{self.args.rounds} over {len(stock_train_loaders)} stocks "
                  f"({epochs_per_stock} epochs/stock)")

            train_losses = []
            for idx, stock_loader in enumerate(stock_train_loaders):
                stock_t = time.time()
                stock_losses = []
                for ep in range(epochs_per_stock):
                    loss = self.train_epoch(stock_loader)
                    stock_losses.append(loss)
                avg_loss = sum(stock_losses) / len(stock_losses)
                train_losses.append(avg_loss)
                if idx % 50 == 0:
                    print(f"  Stock {idx}/{len(stock_train_loaders)} | "
                          f"Avg Loss: {avg_loss:.5f} | Time: {time.time()-stock_t:.1f}s")

            train_loss = sum(train_losses) / len(train_losses)
            
            val_metrics = evaluate(self.model, val_loader, self.device, self.criterion)
            val_loss = val_metrics["loss"]
            
            print(f"Round: {r+1} | Train Loss: {train_loss:.5f} Vali Loss: {val_loss:.5f} | Time: {time.time()-t1:.2f}s")
            
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model at round {r+1}")
                
            adjust_learning_rate(self.optimizer, r + 1, self.args)
            
        self.model.load_state_dict(torch.load(save_path))
        return self.model
