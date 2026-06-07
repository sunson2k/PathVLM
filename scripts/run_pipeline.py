#!/usr/bin/env python
"""Master orchestrator: run full pipeline end-to-end."""

import sys
import os
import subprocess
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_script(script_name: str, script_path: str) -> int:
    """
    Run a Python script and return exit code.
    
    Args:
        script_name: Name of script (for logging)
        script_path: Path to script
    
    Returns:
        Exit code (0 = success)
    """
    logger.info("=" * 70)
    logger.info(f"RUNNING: {script_name}")
    logger.info("=" * 70)
    
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))
    
    if result.returncode != 0:
        logger.error(f"✗ {script_name} FAILED (exit code: {result.returncode})")
        return result.returncode
    
    logger.info(f"✓ {script_name} COMPLETED")
    logger.info("")
    
    return 0


def main():
    """Run full pipeline."""
    
    logger.info("=" * 70)
    logger.info("PATHVLM: FULL PIPELINE ORCHESTRATOR")
    logger.info("=" * 70)
    logger.info("")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define pipeline steps
    pipeline_steps = [
        ("01 - Data Preparation", os.path.join(script_dir, "01_prepare_data.py")),
        ("02 - Train Image Model", os.path.join(script_dir, "02_train_image.py")),
        ("03 - Train Visual Model", os.path.join(script_dir, "03_train_visual.py")),
        ("04 - Train Multimodal Model", os.path.join(script_dir, "04_train_multimodal.py")),
    ]
    
    # Execute pipeline
    for step_name, step_path in pipeline_steps:
        exit_code = run_script(step_name, step_path)
        if exit_code != 0:
            logger.error("")
            logger.error("=" * 70)
            logger.error(f"PIPELINE FAILED at step: {step_name}")
            logger.error("=" * 70)
            return 1
    
    # Pipeline complete
    logger.info("=" * 70)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Results saved to: {Config.paths.results_dir}")
    logger.info("  - image_mode/")
    logger.info("  - visual_mode/")
    logger.info("  - multimodal_mode/")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
