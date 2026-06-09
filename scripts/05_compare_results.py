#!/usr/bin/env python
"""Script 05: Compare evaluated model outputs and generate an HTML report."""

import sys
import os

# Add project root to path so src can be imported as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.visualization import plot_loss_curves, generate_comparison_report
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
        mode: os.path.join(result_dir, 'checkpoints', 'history.json')
        for mode, result_dir in Config.paths.model_results_dirs.items()
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
    results_dirs = Config.paths.model_results_dirs
    
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
