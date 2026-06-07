#!/usr/bin/env python
"""Script 01: Data preparation - split and align samples across all modalities."""

import sys
import os

# Add project root to path so src can be imported as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preparation import DataPreparer
from src.config import Config, setup_directories
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run data preparation pipeline."""
    
    # Create output directories
    setup_directories()
    
    # Initialize preparer
    preparer = DataPreparer(config=Config)
    
    # Run full pipeline
    train_df, val_df, test_df = preparer.run_full_pipeline()
    
    logger.info("✓ Data preparation complete!")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = main()
