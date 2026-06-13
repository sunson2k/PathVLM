#!/usr/bin/env python
"""Script 04: Train multimodal embedding model."""

import argparse
import logging
import os
import sys

# Add project root to path so src can be imported as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config, setup_directories
from src.data_loaders import create_dataloaders
from src.evaluation import evaluate_all_splits
from src.models import MultimodalDNN, MultimodalCrossAttentionDNN, MultimodalGMUDNN
from src.training import Trainer, resolve_device

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the multimodal model.')
    parser.add_argument(
        '--split-dir',
        default=Config.paths.data_splits_dir,
        help='Directory containing train_split.csv, val_split.csv, and test_split.csv.',
    )
    parser.add_argument(
        '--results-suffix',
        default='',
        help='Optional subdirectory under results/ for this run, e.g. fold_0.',
    )
    return parser.parse_args()


def result_dir(results_suffix: str) -> str:
    suffix = results_suffix.strip("/\\")
    if suffix:
        return os.path.join(Config.paths.results_dir, suffix, Config.paths.multimodal_results_name)
    return Config.paths.multimodal_results_dir


def create_multimodal_model(num_genes: int):
    """Create the configured multimodal architecture."""
    model_type = Config.model.multimodal_model.lower()
    common_kwargs = {
        "num_genes": num_genes,
        "hidden_dims": Config.model.dnn_hidden_sizes,
        "dropout": Config.model.dnn_dropout,
        "normalization": Config.model.dnn_normalization,
    }

    if model_type == "concat":
        logger.info("Using multimodal architecture: concat")
        return MultimodalDNN(**common_kwargs)

    if model_type == "cross_attention":
        logger.info("Using multimodal architecture: cross_attention")
        return MultimodalCrossAttentionDNN(**common_kwargs)

    if model_type == "gmu":
        logger.info("Using multimodal architecture: gmu")
        return MultimodalGMUDNN(**common_kwargs)

    raise ValueError(
        "Unsupported model.multimodal_model "
        f"'{Config.model.multimodal_model}'. Use 'concat', 'cross_attention', or 'gmu'."
    )


def main(split_dir: str = None, results_suffix: str = ''):
    """Train multimodal (visual + text) embedding-based DNN model."""

    # Setup
    setup_directories()
    device = resolve_device(Config.training.device)
    split_dir = split_dir or Config.paths.data_splits_dir
    output_root = result_dir(results_suffix)

    # Create dataloaders
    logger.info("Creating dataloaders for multimodal mode...")
    logger.info(f"Using split directory: {split_dir}")
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        split_dir=split_dir,
        feature_mode="multimodal",
        batch_size=Config.training.batch_size,
        num_workers=Config.training.num_workers,
        expr_csv_dir=Config.data.expr_dir,
        data_root=Config.data.data_root,
    )

    # Get gene names for evaluation
    import pandas as pd

    expr_root = os.path.join(
        Config.data.data_root, Config.data.tissue, Config.data.expr_dir
    )
    expr_df = pd.read_csv(
        os.path.join(expr_root, os.listdir(expr_root)[0]), index_col=0
    )
    gene_names = expr_df.columns.str.strip().tolist()

    model = create_multimodal_model(num_genes=len(gene_names))

    # Create trainer
    checkpoint_dir = os.path.join(output_root, "checkpoints")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=Config.training.learning_rate,
        weight_decay=Config.training.weight_decay,
        loss_eps=Config.training.loss_eps,
        checkpoint_dir=checkpoint_dir,
        mode_name="multimodal",
    )

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        max_epochs=Config.training.max_epochs,
        early_stop_patience=Config.training.early_stop_patience,
    )

    # Save history
    trainer.save_history()

    # Evaluate
    logger.info("Evaluating on all splits...")
    eval_dir = os.path.join(output_root, "evaluation")
    results = evaluate_all_splits(
        model=trainer.model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        gene_names=gene_names,
        device=device,
        loss_eps=Config.training.loss_eps,
        output_dir=eval_dir,
        mode_name="multimodal",
    )

    logger.info("✓ Multimodal mode training complete!")
    logger.info(f"  Best model saved to: {checkpoint_dir}")
    logger.info(f"  Evaluation results saved to: {eval_dir}")

    return trainer, results


if __name__ == "__main__":
    args = parse_args()
    trainer, results = main(split_dir=args.split_dir, results_suffix=args.results_suffix)
