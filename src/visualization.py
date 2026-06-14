"""Visualization: loss trajectories and comparison plots."""
import os
import json
import matplotlib.pyplot as plt
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def plot_loss_curves(history_dirs: Dict[str, str], 
                     output_dir: str = None,
                     save_name: str = "loss_comparison.png",
                     y_min: float = None,
                     y_max: float = None) -> str:
    """
    Plot loss curves for all modes (comparison view: train/val/test splits with all models).
    
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
    
    color_map = plt.get_cmap('tab10')
    colors = {
        mode_name: color_map(idx % color_map.N)
        for idx, mode_name in enumerate(history_dirs)
    }
    
    for mode_name, history_path in history_dirs.items():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        color = colors[mode_name]
        
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
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, save_name)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved loss curves: {output_path}")
    plt.close()
    
    return output_path


def plot_per_model_loss_curves(history_dirs: Dict[str, str],
                               output_dir: str = None,
                               save_name: str = "loss_per_model.png",
                               y_min: float = None,
                               y_max: float = None) -> str:
    """
    Plot loss curves for each model individually (per-model view: each model has 3 curves).
    
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
    
    mode_names = list(history_dirs.keys())
    fig_width = max(6, min(24, 6 * len(mode_names)))
    fig, axes = plt.subplots(1, len(mode_names), figsize=(fig_width, 5))
    if len(mode_names) == 1:
        axes = [axes]

    loss_colors = {'train': 'red', 'val': 'orange', 'test': 'green'}
    
    for idx, mode_name in enumerate(mode_names):
        history_path = history_dirs.get(mode_name)
        if not history_path:
            continue
            
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Plot train, val, test for this mode
        axes[idx].plot(history['train_loss'], label='Train', 
                      color=loss_colors['train'], linewidth=2)
        axes[idx].plot(history['val_loss'], label='Validation', 
                      color=loss_colors['val'], linewidth=2)
        axes[idx].plot(history['test_loss'], label='Test', 
                      color=loss_colors['test'], linewidth=2)
        
        axes[idx].set_title(mode_name, fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel('Loss', fontsize=12)
        if y_min is not None or y_max is not None:
            axes[idx].set_ylim(bottom=y_min, top=y_max)
        axes[idx].legend(fontsize=11)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_path = os.path.join(output_dir, save_name)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved per-model loss curves: {output_path}")
    plt.close()
    
    return output_path



def _split_fold_model_name(setup_name: str) -> tuple[str, str]:
    """Split a discovered setup name into fold and model labels."""
    parts = setup_name.split('/')
    if len(parts) >= 2 and parts[0].startswith('fold_'):
        return parts[0], parts[1]
    return 'single', setup_name


def _model_display_name(model_name: str) -> str:
    """Map result folder names to stable report labels."""
    if model_name.startswith('image'):
        return 'image'
    if model_name.startswith('visual'):
        return 'visual'
    if model_name.startswith('multimodal'):
        return 'multimodal'
    return model_name


def plot_fold_loss_comparison(history_dirs: Dict[str, str],
                              output_dir: str = None,
                              save_name: str = "loss_comparison.png",
                              y_min: float = None,
                              y_max: float = None) -> str:
    """
    Plot one model-comparison row per fold and train/val/test columns.

    Args:
        history_dirs: Dict mapping discovered setup name to history JSON path.
        output_dir: Directory to save plot.
        save_name: Filename for saved plot.
        y_min: Optional lower Y-axis bound.
        y_max: Optional upper Y-axis bound.

    Returns:
        Path to saved plot.
    """
    if output_dir is None:
        output_dir = os.path.dirname(next(iter(history_dirs.values())))

    os.makedirs(output_dir, exist_ok=True)

    grouped = {}
    for setup_name, history_path in history_dirs.items():
        fold_name, model_name = _split_fold_model_name(setup_name)
        model_label = _model_display_name(model_name)
        grouped.setdefault(fold_name, {})[model_label] = history_path

    fold_names = sorted(grouped)
    split_keys = [('train_loss', 'Training Loss'), ('val_loss', 'Validation Loss'), ('test_loss', 'Test Loss')]
    model_order = ['image', 'visual', 'multimodal']

    fig_height = max(4, 3.6 * len(fold_names))
    fig, axes = plt.subplots(len(fold_names), 3, figsize=(18, fig_height), squeeze=False)
    color_map = {'image': 'tab:blue', 'visual': 'tab:orange', 'multimodal': 'tab:green'}

    for row_idx, fold_name in enumerate(fold_names):
        histories = {}
        for model_label, history_path in grouped[fold_name].items():
            with open(history_path, 'r') as f:
                histories[model_label] = json.load(f)

        for col_idx, (loss_key, title) in enumerate(split_keys):
            ax = axes[row_idx][col_idx]
            for model_label in model_order:
                history = histories.get(model_label)
                if not history or loss_key not in history:
                    continue
                ax.plot(
                    history[loss_key],
                    label=model_label,
                    color=color_map.get(model_label),
                    linewidth=2,
                )
            ax.set_title(f'{fold_name} - {title}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            if y_min is not None or y_max is not None:
                ax.set_ylim(bottom=y_min, top=y_max)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(output_dir, save_name)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved fold loss comparison curves: {output_path}")
    plt.close()

    return output_path
