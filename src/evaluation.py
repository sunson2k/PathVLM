"""Evaluation and metrics computation."""
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import json
import logging
from typing import Dict, List, Tuple

from models import scaled_mse_loss

logger = logging.getLogger(__name__)


class Evaluator:
    """Compute and track evaluation metrics."""
    
    def __init__(self, model, device: str = "cuda"):
        """
        Args:
            model: PyTorch model
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.device = device
    
    @torch.no_grad()
    def compute_metrics(self, 
                       dataloader: DataLoader,
                       gene_names: List[str],
                       split_name: str = "val") -> Dict:
        """
        Compute comprehensive metrics.
        
        Args:
            dataloader: PyTorch DataLoader
            gene_names: List of gene names
            split_name: Name of split (for logging)
        
        Returns:
            Dictionary with metrics:
                - mse_loss: Mean squared error (scaled)
                - per_gene_mse: Dict[gene_name -> mse]
                - per_gene_pearson: Dict[gene_name -> pearson_r]
                - per_gene_pearson_pval: Dict[gene_name -> p_value]
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_uids = []
        
        total_loss = 0.0
        
        for features, targets, uids in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(features)
            loss = scaled_mse_loss(predictions, targets)
            total_loss += loss.item() * features.size(0)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_uids.extend(uids)
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0).numpy()  # (N, num_genes)
        all_targets = torch.cat(all_targets, dim=0).numpy()           # (N, num_genes)
        
        avg_loss = total_loss / len(dataloader.dataset)
        
        # Compute per-gene metrics
        per_gene_mse = {}
        per_gene_pearson = {}
        per_gene_pearson_pval = {}
        
        for idx, gene_name in enumerate(gene_names):
            pred = all_predictions[:, idx]
            true = all_targets[:, idx]
            
            # MSE
            mse = np.mean((pred - true) ** 2)
            per_gene_mse[gene_name] = float(mse)
            
            # Pearson correlation
            try:
                r, pval = pearsonr(true, pred)
                per_gene_pearson[gene_name] = float(r)
                per_gene_pearson_pval[gene_name] = float(pval)
            except Exception as e:
                logger.warning(f"Could not compute correlation for {gene_name}: {e}")
                per_gene_pearson[gene_name] = np.nan
                per_gene_pearson_pval[gene_name] = np.nan
        
        metrics = {
            'split': split_name,
            'num_samples': len(dataloader.dataset),
            'mse_loss': float(avg_loss),
            'per_gene_mse': per_gene_mse,
            'per_gene_pearson': per_gene_pearson,
            'per_gene_pearson_pval': per_gene_pearson_pval,
            'mean_gene_mse': float(np.mean(list(per_gene_mse.values()))),
            'median_gene_mse': float(np.median(list(per_gene_mse.values()))),
            'mean_pearson': float(np.nanmean(list(per_gene_pearson.values()))),
            'median_pearson': float(np.nanmedian(list(per_gene_pearson.values())))
        }
        
        return metrics, all_predictions, all_targets, all_uids
    
    def save_predictions(self, 
                        predictions: np.ndarray,
                        targets: np.ndarray,
                        uids: List[str],
                        gene_names: List[str],
                        output_path: str):
        """
        Save predictions and targets to CSV.
        
        Args:
            predictions: (N, num_genes) array
            targets: (N, num_genes) array
            uids: List of sample identifiers
            gene_names: List of gene names
            output_path: Path to save CSV
        """
        # Create DataFrame with predictions
        data = {'uid': uids}
        for idx, gene_name in enumerate(gene_names):
            data[f'pred_{gene_name}'] = predictions[:, idx]
            data[f'true_{gene_name}'] = targets[:, idx]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions: {output_path}")
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """Save metrics as JSON."""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {output_path}")


def evaluate_all_splits(model,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       test_loader: DataLoader,
                       gene_names: List[str],
                       device: str = "cuda",
                       output_dir: str = None,
                       mode_name: str = "model") -> Dict[str, Dict]:
    """
    Evaluate model on all splits.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        gene_names: List of gene names
        device: Device to evaluate on
        output_dir: Directory to save results
        mode_name: Name of mode (for logging/saving)
    
    Returns:
        Dictionary with metrics for each split
    """
    evaluator = Evaluator(model, device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    logger.info(f"Evaluating {mode_name} model on all splits...")
    
    for split_name, dataloader in [('train', train_loader), 
                                   ('val', val_loader), 
                                   ('test', test_loader)]:
        logger.info(f"  Evaluating on {split_name} split...")
        
        metrics, predictions, targets, uids = evaluator.compute_metrics(
            dataloader, gene_names, split_name=split_name
        )
        
        results[split_name] = metrics
        
        # Save predictions
        pred_path = os.path.join(output_dir, f'{split_name}_predictions.csv')
        evaluator.save_predictions(predictions, targets, uids, gene_names, pred_path)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f'{split_name}_metrics.json')
        evaluator.save_metrics(metrics, metrics_path)
        
        logger.info(f"    {split_name.upper()} Loss: {metrics['mse_loss']:.6f}")
        logger.info(f"    Mean Gene Pearson: {metrics['mean_pearson']:.4f}")
    
    # Save aggregated results
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")
    
    return results
