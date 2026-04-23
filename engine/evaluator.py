import torch
import numpy as np
from utils.metrics import metric

def evaluate(model, test_loader, device, criterion=None):
    model.eval()
    preds = []
    trues = []
    total_loss = 0
    
    if criterion is None:
        criterion = torch.nn.MSELoss()
        
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Forward pass
            # Note: AdaPatch returns a tuple (pred, slice, decode), we only need pred
            outputs = model(batch_x, None)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    mae, mse, rmse, r2 = metric(preds, trues)
    
    return {
        "loss": total_loss / len(test_loader),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
