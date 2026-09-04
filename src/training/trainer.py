import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import os
import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None # type: ignore

class Trainer:
    """
    Two-phase training loop for GNN Reservoir Prediction system.
    """
    def __init__(self, model: nn.Module, criterion: nn.Module, edge_index: torch.Tensor, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', log_dir: str = 'runs'):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The GNN model.
            criterion (nn.Module): The loss function.
            edge_index (torch.Tensor): The graph edge connectivity.
            device (str): Device to train on.
            log_dir (str): Directory for TensorBoard logs.
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.edge_index = edge_index.to(device)
        self.writer = SummaryWriter(log_dir) if SummaryWriter else None
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda')) # type: ignore

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, clip_grad: float = 1.0) -> float:
        """
        Trains the model for one epoch.

        Args:
            dataloader (DataLoader): Training dataloader.
            optimizer (torch.optim.Optimizer): Optimizer.
            clip_grad (float): Maximum gradient norm.

        Returns:
            float: Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        pbar = tqdm.tqdm(dataloader, desc="Training")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')): # type: ignore
                preds = self.model(batch['node_features'], batch['climate_indices'], self.edge_index)
                loss = self.criterion(preds, batch['targets'], batch['oni'])

            self.scaler.scale(loss).backward() # type: ignore
            self.scaler.unscale_(optimizer) # type: ignore
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.scaler.step(optimizer) # type: ignore
            self.scaler.update() # type: ignore

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validates the model.

        Args:
            dataloader (DataLoader): Validation dataloader.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')): # type: ignore
                    preds = self.model(batch['node_features'], batch['climate_indices'], self.edge_index)
                    loss = self.criterion(preds, batch['targets'], batch['oni'])
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, save_dir: str, phase: str = "pretrain", patience: int = 10, weight_decay: float = 0.01):
        """
        Runs the training loop.

        Args:
            train_loader (DataLoader): Training dataloader.
            val_loader (DataLoader): Validation dataloader.
            epochs (int): Number of epochs.
            lr (float): Learning rate.
            save_dir (str): Directory to save checkpoints.
            phase (str): 'pretrain' or 'finetune'.
            patience (int): Early stopping patience.
            weight_decay (float): L2 regularization weight decay.
        """
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss = self.validate(val_loader)
            scheduler.step()
            
            if self.writer:
                self.writer.add_scalar(f'Loss/train_{phase}', train_loss, epoch)
                self.writer.add_scalar(f'Loss/val_{phase}', val_loss, epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_dir, f'best_model_{phase}.pt'))
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def run_two_phase_training(self, synthetic_train_loader: DataLoader, synthetic_val_loader: DataLoader, 
                               obs_train_loader: DataLoader, obs_val_loader: DataLoader, 
                               save_dir: str, weight_decay: float = 0.01):
        """
        Executes the two-phase training loop.

        Args:
            synthetic_train_loader (DataLoader): Phase 1 train dataloader.
            synthetic_val_loader (DataLoader): Phase 1 validation dataloader.
            obs_train_loader (DataLoader): Phase 2 train dataloader.
            obs_val_loader (DataLoader): Phase 2 validation dataloader.
            save_dir (str): Directory to save checkpoints.
            weight_decay (float): L2 regularization weight decay.
        """
        print("Starting Phase 1: Pre-training on synthetic data...")
        self.train(synthetic_train_loader, synthetic_val_loader, epochs=50, lr=1e-3, save_dir=save_dir, phase="pretrain", patience=10, weight_decay=weight_decay)
        
        print("Loading best pre-trained model for fine-tuning...")
        self.model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model_pretrain.pt')))
        
        print("Starting Phase 2: Fine-tuning on observed data...")
        self.train(obs_train_loader, obs_val_loader, epochs=50, lr=1e-4, save_dir=save_dir, phase="finetune", patience=10, weight_decay=weight_decay)
