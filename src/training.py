"""Training utilities: train/eval loops, early stopping, checkpointing."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from .models import scaled_mse_loss

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str) -> str:
    """Resolve and validate the configured training device."""
    requested_device = (requested_device or "cuda").lower()

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for training, but this PyTorch install cannot "
                "access CUDA. Install a CUDA-enabled torch/torchvision pair or set "
                "training.device to 'cpu' in configs/run_config.json."
            )

        device = requested_device if ":" in requested_device else "cuda:0"
        device_index = torch.device(device).index or 0
        gpu_name = torch.cuda.get_device_name(device_index)
        total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        logger.info(f"Using device: {device} ({gpu_name}, {total_gb:.1f} GB)")
        return device

    logger.info(f"Using device: {requested_device}")
    return requested_device


class EarlyStopper:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 2):
        """
        Args:
            patience: Number of epochs with no improvement to wait before stopping
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered (patience={self.patience})")
                logger.info(f"Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
                return True
            return False


class Trainer:
    """Training orchestrator."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str = "cuda",
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-4,
                 checkpoint_dir: str = None,
                 mode_name: str = "model"):
        """
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            checkpoint_dir: Directory to save checkpoints
            mode_name: Name of mode for logging (e.g., 'image', 'visual', 'multimodal')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.mode_name = mode_name
        
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Create checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = f"checkpoints_{mode_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
    
    def train_epoch(self) -> float:
        """Train one epoch. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (features, targets, _) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = scaled_mse_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                logger.debug(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                           f"Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on a dataloader. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        
        for features, targets, _ in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(features)
            loss = scaled_mse_loss(predictions, targets)
            total_loss += loss.item() * features.size(0)
        
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        filename = f"checkpoint_epoch_{epoch}.pt" if not is_best else "best_model.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        
        logger.debug(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str, restore_history: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if restore_history:
            self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint: {path}")
    
    def train(self, max_epochs: int = 100, early_stop_patience: int = 2) -> Dict:
        """
        Train the model.
        
        Args:
            max_epochs: Maximum number of epochs
            early_stop_patience: Early stopping patience
        
        Returns:
            Dictionary with training history and final metrics
        """
        logger.info("=" * 60)
        logger.info(f"Starting training ({self.mode_name})")
        logger.info("=" * 60)
        
        early_stopper = EarlyStopper(patience=early_stop_patience)
        
        for epoch in range(1, max_epochs + 1):
            # Train and evaluate
            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.val_loader)
            test_loss = self.evaluate(self.test_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['test_loss'].append(test_loss)
            
            # Log every epoch so terminal and log output match the saved history.
            logger.info(f"Epoch {epoch:3d}/{max_epochs} | "
                      f"Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"Test: {test_loss:.6f}")
            
            # Check for best model and save
            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = epoch
                self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping check
            if early_stopper(val_loss, epoch):
                logger.info(f"Training stopped at epoch {epoch}")
                break
        
        # Load best model
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self.load_checkpoint(best_path, restore_history=False)
        
        logger.info("=" * 60)
        logger.info(f"Training complete ({self.mode_name})")
        logger.info("=" * 60)
        
        return self.history
    
    def save_history(self):
        """Save training history as JSON."""
        path = os.path.join(self.checkpoint_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved history: {path}")
