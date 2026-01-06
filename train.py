import os
import sys
import json
import yaml
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# 1. HPC PRE-REQUISITE: Use non-interactive backend for plots
import matplotlib
matplotlib.use('Agg') 

# 2. Setup Paths (Mirroring your sys.path logic)
p1 = Path.cwd()
p2 = Path.cwd().parent
csdi_dir = p2 / "CSDI"
sys.path.insert(0, str(csdi_dir))
sys.path.insert(0, str(p1 / "data"))

from main_model import CSDI_base
from yahoo_data import get_dataloader, FinancialDataset

# 3. Custom CSDI Financial Class (from Cell 6)
class CSDI_Financial(CSDI_base):
    def __init__(self, config, device, target_dim=5):
        super(CSDI_Financial, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        return (observed_data, observed_mask, observed_tp, gt_mask, observed_mask, 
                torch.zeros(len(observed_data)).long().to(self.device))

def run_experiment():
    # 4. Global Variables (from Cell 2)
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2024-12-31"
    seq_len = 252
    val_rat = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5. Load Config & Dataloaders
    with open("configs/base_csdi.yaml", "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(f'data/{ticker}_{start_date}_{end_date}_processed.csv')
    
    full_dataset = FinancialDataset(df, seq_len=seq_len)
    total_len = len(full_dataset)
    val_len = int(total_len * val_rat)
    train_len = total_len - val_len
    
    train_dataset = Subset(full_dataset, range(0, train_len))
    val_dataset = Subset(full_dataset, range(train_len, total_len))
    
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    # 6. Initialize Model
    model = CSDI_Financial(config, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])
    history = {"train_loss": [], "val_loss": [], "volatility_error": []}

    # 7. Training Loop (Optimized for HPC logging)
    model.train()
    for epoch in range(config["train"]["epochs"]):
        cumulative_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch, is_train=1) 
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item()
            if i >= config["train"]["itr_per_epoch"]: break
                
        avg_loss = cumulative_loss / (i + 1)
        history['train_loss'].append(avg_loss)

        # Validation Logic
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_loss = model(val_batch, is_train=0)
            history["val_loss"].append(val_loss.item())

            samples, observed, _, _, _ = model.evaluate(val_batch, n_samples=5)
            vol_err = abs(samples.std().item() - observed.std().item())
            history["volatility_error"].append(vol_err)

        print(f"Epoch {epoch}: Train Loss {avg_loss:.4f}, Val Loss {val_loss.item():.4f}, Vol Err {vol_err:.4f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"../checkpoints/checkpoint_epoch_{epoch}.pth")

    # 8. Save History & Final Model
    with open(f"../checkpoints/{ticker}_history.json", 'w') as f:
        json.dump(history, f)
    torch.save(model.state_dict(), f"../checkpoints/final_model.pth")

if __name__ == "__main__":
    run_experiment()