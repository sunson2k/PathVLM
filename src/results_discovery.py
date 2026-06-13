"""Helpers for finding completed result setups."""
from pathlib import Path
from typing import Dict, Iterable

from src.config import Config


def default_results_dir() -> Path:
    """Return the active experiment results directory used by report generation."""
    return Path(Config.paths.results_dir)


def discover_result_dirs(results_dir: Path, exclude_names: Iterable[str] = ("old",)) -> Dict[str, str]:
    """Return result setup directories with evaluation summaries, excluding archive folders."""
    results_dir = Path(results_dir)
    excluded = set(exclude_names)
    discovered = {}

    if not results_dir.exists():
        return {}

    for summary_path in results_dir.rglob("evaluation/summary.json"):
        try:
            relative_parts = summary_path.relative_to(results_dir).parts
        except ValueError:
            continue
        if any(part in excluded for part in relative_parts):
            continue

        result_dir = summary_path.parent.parent
        setup_name = result_dir.relative_to(results_dir).as_posix()
        discovered[setup_name] = str(result_dir)

    return {
        setup_name: discovered[setup_name]
        for setup_name in sorted(discovered)
    }


def discover_history_files(results_dir: Path, exclude_names: Iterable[str] = ("old",)) -> Dict[str, str]:
    """Return checkpoint histories for discovered result setup directories."""
    history_files = {}
    for setup_name, result_dir in discover_result_dirs(results_dir, exclude_names).items():
        history_path = Path(result_dir) / "checkpoints" / "history.json"
        if history_path.exists():
            history_files[setup_name] = str(history_path)
    return history_files
