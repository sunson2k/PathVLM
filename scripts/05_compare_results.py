#!/usr/bin/env python
"""Script 05: Compare evaluated model outputs and generate an HTML report."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from visualization import plot_loss_curves, generate_comparison_report
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Generate comparison visualizations and reports."""
    
    logger.info("=" * 60)
    logger.info("GENERATING COMPARISON REPORT")
    logger.info("=" * 60)
    
    results_dir = Config.paths.results_dir
    
    # Paths to history files for each mode
    history_dirs = {
        'image': os.path.join(results_dir, 'image_mode', 'checkpoints', 'history.json'),
        'visual': os.path.join(results_dir, 'visual_mode', 'checkpoints', 'history.json'),
        'multimodal': os.path.join(results_dir, 'multimodal_mode', 'checkpoints', 'history.json')
    }
    
    # Filter to only existing files
    history_dirs = {k: v for k, v in history_dirs.items() if os.path.exists(v)}
    
    if history_dirs:
        # Plot loss curves
        logger.info("Plotting loss curves...")
        plot_loss_curves(
            history_dirs=history_dirs,
            output_dir=results_dir,
            save_name="loss_comparison.png"
        )
    else:
        logger.warning("No history files found - skipping loss curves")
    
    # Generate comparison report
    logger.info("Generating comparison report...")
    results_dirs = {
        'image': os.path.join(results_dir, 'image_mode'),
        'visual': os.path.join(results_dir, 'visual_mode'),
        'multimodal': os.path.join(results_dir, 'multimodal_mode')
    }
    
    report_path = generate_comparison_report(
        results_dirs=results_dirs,
        output_path=os.path.join(results_dir, 'comparison_report.html')
    )
    
    logger.info("=" * 60)
    logger.info("✓ COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Outputs:")
    logger.info(f"  Loss Curves: {os.path.join(results_dir, 'loss_comparison.png')}")
    logger.info(f"  Report: {report_path}")
    logger.info("")


if __name__ == "__main__":
    main()
