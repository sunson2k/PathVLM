#!/scr2/lucasni/.venv/bin/python
"""Script 03: Train visual embedding model."""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from data_loaders import create_dataloaders
from models import VisualDNN
from training import Trainer
from evaluation import evaluate_all_splits
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Train visual embedding-based DNN model."""
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders for visual embedding mode...")
    split_dir = Config.paths.data_splits_dir
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        split_dir=split_dir,
        feature_mode='visual_embedding',
        batch_size=Config.training.batch_size,
        num_workers=Config.training.num_workers,
        expr_csv_dir='ST-expression-top-8n',
        data_root=Config.data.data_root
    )
    
    # Get gene names for evaluation
    import pandas as pd
    expr_df = pd.read_csv(
        os.path.join(Config.data.data_root, Config.data.tissue, 'ST-expression-top-8n', 
                     os.listdir(os.path.join(Config.data.data_root, Config.data.tissue, 'ST-expression-top-8n'))[0])
    )
    gene_names = expr_df.columns.tolist()
    
    # Create model
    logger.info("Creating VisualDNN model...")
    model = VisualDNN(
        num_genes=len(gene_names),
        dropout=0.4
    )
    
    # Create trainer
    checkpoint_dir = os.path.join(Config.paths.results_dir, 'visual_mode', 'checkpoints')
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=Config.training.learning_rate,
        weight_decay=Config.training.weight_decay,
        checkpoint_dir=checkpoint_dir,
        mode_name='visual'
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(
        max_epochs=Config.training.max_epochs,
        early_stop_patience=Config.training.early_stop_patience
    )
    
    # Save history
    trainer.save_history()
    
    # Evaluate
    logger.info("Evaluating on all splits...")
    eval_dir = os.path.join(Config.paths.results_dir, 'visual_mode', 'evaluation')
    results = evaluate_all_splits(
        model=trainer.model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        gene_names=gene_names,
        device=device,
        output_dir=eval_dir,
        mode_name='visual'
    )
    
    logger.info("✓ Visual mode training complete!")
    logger.info(f"  Best model saved to: {checkpoint_dir}")
    logger.info(f"  Evaluation results saved to: {eval_dir}")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
