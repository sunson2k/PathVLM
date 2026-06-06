"""Visualization: loss trajectories and comparison plots."""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def plot_loss_curves(history_dirs: Dict[str, str], 
                     output_dir: str = None,
                     save_name: str = "loss_comparison.png") -> str:
    """
    Plot loss curves for all modes.
    
    Args:
        history_dirs: Dict mapping mode name to history JSON path
        output_dir: Directory to save plot
        save_name: Filename for saved plot
    
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = os.path.dirname(next(iter(history_dirs.values())))
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'image': 'blue', 'visual': 'orange', 'multimodal': 'green'}
    
    for mode_name, history_path in history_dirs.items():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        color = colors.get(mode_name, 'black')
        
        # Train loss
        axes[0].plot(history['train_loss'], label=mode_name, color=color, linewidth=2)
        
        # Val loss
        axes[1].plot(history['val_loss'], label=mode_name, color=color, linewidth=2)
        
        # Test loss
        axes[2].plot(history['test_loss'], label=mode_name, color=color, linewidth=2)
    
    # Format subplots
    for idx, (ax, title) in enumerate(zip(axes, ['Training Loss', 'Validation Loss', 'Test Loss'])):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, save_name)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved loss curves: {output_path}")
    plt.close()
    
    return output_path


def generate_comparison_report(results_dirs: Dict[str, str],
                              output_path: str = None) -> str:
    """
    Generate HTML comparison report.
    
    Args:
        results_dirs: Dict mapping mode name to results directory
        output_path: Path to save HTML report
    
    Returns:
        Path to saved HTML file
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(next(iter(results_dirs.values()))), 
                                   "comparison_report.html")
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PathVLM - Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .metric { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>PathVLM: Multi-Modal Gene Expression Prediction - Comparison Report</h1>
    """
    
    # Load metrics for each mode
    metrics_data = {}
    for mode_name, results_dir in results_dirs.items():
        summary_path = os.path.join(results_dir, 'evaluation', 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                metrics_data[mode_name] = json.load(f)
    
    if metrics_data:
        # Add comparison table
        html += """
        <h2>Loss Comparison</h2>
        <table>
            <tr>
                <th>Mode</th>
                <th>Train Loss</th>
                <th>Val Loss</th>
                <th>Test Loss</th>
            </tr>
        """
        
        for mode_name in ['image', 'visual', 'multimodal']:
            if mode_name in metrics_data:
                data = metrics_data[mode_name]
                html += f"""
            <tr>
                <td class='metric'>{mode_name.upper()}</td>
                <td>{data['train']['mse_loss']:.6f}</td>
                <td>{data['val']['mse_loss']:.6f}</td>
                <td>{data['test']['mse_loss']:.6f}</td>
            </tr>
                """
        
        html += "</table>"
        
        # Add Pearson correlation comparison
        html += """
        <h2>Mean Pearson Correlation</h2>
        <table>
            <tr>
                <th>Mode</th>
                <th>Train</th>
                <th>Val</th>
                <th>Test</th>
            </tr>
        """
        
        for mode_name in ['image', 'visual', 'multimodal']:
            if mode_name in metrics_data:
                data = metrics_data[mode_name]
                html += f"""
            <tr>
                <td class='metric'>{mode_name.upper()}</td>
                <td>{data['train'].get('mean_pearson', 'N/A')}</td>
                <td>{data['val'].get('mean_pearson', 'N/A')}</td>
                <td>{data['test'].get('mean_pearson', 'N/A')}</td>
            </tr>
                """
        
        html += "</table>"
    
    html += """
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Saved comparison report: {output_path}")
    return output_path
