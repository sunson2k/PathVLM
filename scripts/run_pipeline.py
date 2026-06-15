#!/usr/bin/env python
"""Master orchestrator: run full pipeline end-to-end."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def project_root() -> str:
    """Return the repository root for resolving default CLI paths."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_config_path(config_path: str) -> str:
    """Resolve config paths from the project root unless absolute."""
    if os.path.isabs(config_path):
        return os.path.abspath(config_path)
    return os.path.abspath(os.path.join(project_root(), config_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full PathVLM pipeline.")
    parser.add_argument(
        "--config",
        default="configs/run_config.json",
        help="Run config JSON file. Relative paths are resolved from the project root.",
    )
    return parser.parse_args()


@dataclass
class PipelineJob:
    """A subprocess job in the pipeline."""

    name: str
    script_path: str
    args: List[str]


def run_script(
    script_name: str,
    script_path: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """
    Run a Python script and return exit code.

    Args:
        script_name: Name of script (for logging)
        script_path: Path to script
        args: Optional script arguments
        env: Optional subprocess environment overrides

    Returns:
        Exit code (0 = success)
    """
    args = args or []
    logger.info("=" * 70)
    logger.info(f"RUNNING: {script_name}")
    logger.info("=" * 70)

    result = subprocess.run(
        [sys.executable, script_path, *args],
        cwd=os.path.dirname(script_path),
        env=env,
    )

    if result.returncode != 0:
        logger.error(f"✗ {script_name} FAILED (exit code: {result.returncode})")
        return result.returncode

    logger.info(f"✓ {script_name} COMPLETED")
    logger.info("")

    return 0


def discover_free_gpus() -> List[str]:
    """Return visible GPU IDs sorted by free memory, descending."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        logger.info("nvidia-smi not found; model trainings will run sequentially.")
        return []

    query = [
        nvidia_smi,
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(query, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("Could not query GPUs with nvidia-smi: %s", exc)
        return []

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        gpu_id, free_mb = parts
        try:
            gpus.append((gpu_id, int(free_mb)))
        except ValueError:
            continue

    gpus.sort(key=lambda item: item[1], reverse=True)
    gpu_ids = [gpu_id for gpu_id, _ in gpus]
    if gpu_ids:
        logger.info("Detected GPUs by free memory: %s", ", ".join(gpu_ids))
    else:
        logger.info("No GPUs reported by nvidia-smi; model trainings will run sequentially.")
    return gpu_ids


def run_training_jobs(
    jobs: List[PipelineJob],
    gpu_ids: List[str],
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Run training jobs in parallel across GPUs, or sequentially without GPUs."""
    if not gpu_ids:
        for job in jobs:
            exit_code = run_script(job.name, job.script_path, job.args, env=env)
            if exit_code != 0:
                return exit_code
        return 0

    pending = list(jobs)
    available_gpus = list(gpu_ids)
    running = []

    while pending or running:
        while pending and available_gpus:
            job = pending.pop(0)
            gpu_id = available_gpus.pop(0)
            job_env = (env or os.environ).copy()
            job_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            job_env["PATHVLM_TRAINING_DEVICE"] = "cuda:0"
            logger.info("Starting %s on GPU %s", job.name, gpu_id)
            process = subprocess.Popen(
                [sys.executable, job.script_path, *job.args],
                cwd=os.path.dirname(job.script_path),
                env=job_env,
            )
            running.append({"job": job, "gpu_id": gpu_id, "process": process})

        time.sleep(5)
        still_running = []
        for item in running:
            returncode = item["process"].poll()
            if returncode is None:
                still_running.append(item)
                continue

            job = item["job"]
            gpu_id = item["gpu_id"]
            available_gpus.append(gpu_id)
            if returncode != 0:
                logger.error(
                    "✗ %s FAILED on GPU %s (exit code: %s)",
                    job.name,
                    gpu_id,
                    returncode,
                )
                for other in running:
                    if other is item:
                        continue
                    process = other["process"]
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                return returncode

            logger.info("✓ %s COMPLETED on GPU %s", job.name, gpu_id)

        running = still_running

    return 0


def training_jobs(script_dir: str, split_dir: str, results_suffix: str) -> List[PipelineJob]:
    """Create the three model training jobs for a split directory."""
    common_args = ["--split-dir", split_dir]
    if results_suffix:
        common_args.extend(["--results-suffix", results_suffix])

    return [
        PipelineJob(
            "02 - Train Image Model",
            os.path.join(script_dir, "02_train_image.py"),
            list(common_args),
        ),
        PipelineJob(
            "03 - Train Visual Model",
            os.path.join(script_dir, "03_train_visual.py"),
            list(common_args),
        ),
        PipelineJob(
            "04 - Train Multimodal Model",
            os.path.join(script_dir, "04_train_multimodal.py"),
            list(common_args),
        ),
    ]


def main():
    """Run full pipeline."""
    args = parse_args()
    source_config_path = resolve_config_path(args.config)
    if not os.path.exists(source_config_path):
        logger.error("Config file not found: %s", source_config_path)
        return 1

    os.environ["PATHVLM_RUN_CONFIG_PATH"] = source_config_path

    from src.config import Config, setup_directories, snapshot_run_config
    from src.results_discovery import default_results_dir, discover_result_dirs

    logger.info("=" * 70)
    logger.info("PATHVLM: FULL PIPELINE ORCHESTRATOR")
    logger.info("=" * 70)
    logger.info("")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    prepare_script = os.path.join(script_dir, "01_prepare_data.py")
    summarize_script = os.path.join(script_dir, "05_summarize_results.py")

    setup_directories()
    golden_config_path = snapshot_run_config()
    pipeline_env = os.environ.copy()
    pipeline_env["PATHVLM_RUN_CONFIG_PATH"] = golden_config_path
    logger.info("Source run config: %s", source_config_path)
    logger.info("Golden run config: %s", golden_config_path)

    exit_code = run_script("01 - Data Preparation", prepare_script, env=pipeline_env)
    if exit_code != 0:
        logger.error("PIPELINE FAILED at step: 01 - Data Preparation")
        return exit_code

    gpu_ids = discover_free_gpus()

    fold_indices = range(Config.data.num_folds) if Config.data.cv_enabled else range(1)
    if Config.data.cv_enabled:
        logger.info("Training all %s prepared folds", Config.data.num_folds)
    else:
        logger.info("cv_enabled=false; training only fold_0 for a quick report")

    for fold_idx in fold_indices:
        fold_name = f"fold_{fold_idx}"
        split_dir = os.path.join(Config.paths.data_splits_dir, fold_name)
        logger.info("=" * 70)
        logger.info("RUNNING MODEL TRAINING FOR %s", fold_name)
        logger.info("=" * 70)
        exit_code = run_training_jobs(
            training_jobs(script_dir, split_dir, fold_name),
            gpu_ids,
            env=pipeline_env,
        )
        if exit_code != 0:
            logger.error("PIPELINE FAILED during %s", fold_name)
            return exit_code

    exit_code = run_script("05 - Summarize Results", summarize_script, env=pipeline_env)
    if exit_code != 0:
        logger.error("PIPELINE FAILED at step: 05 - Summarize Results")
        return exit_code

    # Pipeline complete
    logger.info("=" * 70)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    results_dir = default_results_dir()
    logger.info(f"Results saved to: {results_dir}")
    for setup_name in discover_result_dirs(results_dir):
        logger.info(f"  - {setup_name}/")
    logger.info("  - summary.md")
    logger.info("  - loss_comparison.png")
    logger.info("  - loss_per_model.png")
    logger.info("")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
