#!/usr/bin/env python3
"""Generate the PathVLM markdown summary and fold-wise loss comparison plots."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import setup_directories
from src.results_discovery import default_results_dir, discover_history_files, discover_result_dirs
from src.visualization import plot_fold_loss_comparison

MODEL_COLUMNS = ['image', 'visual', 'multimodal']


def load_summary_metrics(result_dir: Path) -> dict:
    summary_path = result_dir / 'evaluation' / 'summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary file: {summary_path}')
    with open(summary_path, 'r') as f:
        return json.load(f)


def _format_metric(value, digits: int) -> str:
    if value is None:
        return 'N/A'
    return f'{value:.{digits}f}'


def _format_count(value) -> str:
    if value is None:
        return 'N/A'
    return str(int(value))


def split_setup_name(setup_name: str) -> tuple[str, str]:
    """Return (fold, model folder) from a discovered result setup name."""
    parts = setup_name.split('/')
    if len(parts) >= 2 and parts[0].startswith('fold_'):
        return parts[0], parts[1]
    return 'single', parts[-1]


def model_column_name(model_folder: str) -> str:
    """Map result folder names to stable report columns."""
    if model_folder.startswith('image'):
        return 'image'
    if model_folder.startswith('visual'):
        return 'visual'
    if model_folder.startswith('multimodal'):
        return 'multimodal'
    return model_folder


def fold_sort_key(fold_name: str):
    if fold_name.startswith('fold_'):
        try:
            return (0, int(fold_name.split('_', 1)[1]))
        except ValueError:
            pass
    return (1, fold_name)


def collect_fold_metrics(results_dirs: dict[str, str]) -> dict:
    """Collect test metrics as fold -> model -> metrics."""
    fold_metrics = {}
    for setup_name, result_dir in results_dirs.items():
        fold_name, model_folder = split_setup_name(setup_name)
        model_name = model_column_name(model_folder)
        if model_name not in MODEL_COLUMNS:
            continue

        summary = load_summary_metrics(Path(result_dir))
        test_summary = summary['test']
        fold_metrics.setdefault(fold_name, {})[model_name] = {
            'mse_loss': test_summary.get('mse_loss'),
            'mean_pearson': test_summary.get('mean_pearson'),
            'num_samples': test_summary.get('num_samples'),
        }
    return fold_metrics


def fold_sample_count(model_metrics: dict) -> int | None:
    """Return the test sample count for a fold, using any available model summary."""
    counts = [metrics.get('num_samples') for metrics in model_metrics.values() if metrics.get('num_samples') is not None]
    if not counts:
        return None
    return max(counts)


def weighted_average(fold_metrics: dict, metric_name: str, model_name: str) -> float | None:
    """Weighted average over folds for one model and metric, weighted by model test samples."""
    numerator = 0.0
    denominator = 0
    for model_metrics in fold_metrics.values():
        metrics = model_metrics.get(model_name)
        if not metrics:
            continue
        value = metrics.get(metric_name)
        weight = metrics.get('num_samples')
        if value is None or weight is None:
            continue
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def weighted_sample_total(fold_metrics: dict) -> int | None:
    """Return total test samples across fold rows, counting each fold once."""
    total = 0
    for model_metrics in fold_metrics.values():
        count = fold_sample_count(model_metrics)
        if count is None:
            continue
        total += count
    return total if total else None


def build_metric_table(fold_metrics: dict, metric_name: str, digits: int) -> list[str]:
    """Build a markdown table for one metric."""
    header = '| Fold | # test patches | image | visual | multimodal |'
    separator = '|---|---:|---:|---:|---:|'
    lines = [header, separator]

    for fold_name in sorted(fold_metrics, key=fold_sort_key):
        model_metrics = fold_metrics[fold_name]
        cells = [fold_name, _format_count(fold_sample_count(model_metrics))]
        for model_name in MODEL_COLUMNS:
            cells.append(_format_metric(model_metrics.get(model_name, {}).get(metric_name), digits))
        lines.append('| ' + ' | '.join(cells) + ' |')

    avg_cells = ['Weighted avg', _format_count(weighted_sample_total(fold_metrics))]
    for model_name in MODEL_COLUMNS:
        avg_cells.append(_format_metric(weighted_average(fold_metrics, metric_name, model_name), digits))
    lines.append('| ' + ' | '.join(avg_cells) + ' |')
    return lines


def build_markdown(
    fold_metrics: dict,
    comparison_img_path: Path | None,
    output_path: Path,
    tissue: str | None,
    expr_name: str | None,
    y_min: float | None,
    y_max: float | None,
):
    title_detail = ''
    if tissue and expr_name:
        title_detail = f' ({tissue} {expr_name})'
    elif tissue:
        title_detail = f' ({tissue})'
    elif expr_name:
        title_detail = f' ({expr_name})'

    y_axis_note = (
        f'- Plot Y-axis ranges are fixed to {y_min if y_min is not None else "auto"}'
        f' to {y_max if y_max is not None else "auto"} for this generated view.'
        if y_min is not None or y_max is not None
        else '- Plot Y-axis ranges are selected independently per subplot for readability.'
    )

    lines = [
        f'# PathVLM Results Summary{title_detail}',
        '',
        f'_Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_',
        '',
        '## 1. Test set final results comparison across discovered result folders',
        '',
        '### Test MSE',
        '',
        *build_metric_table(fold_metrics, 'mse_loss', 6),
        '',
        '### Test Pearson',
        '',
        *build_metric_table(fold_metrics, 'mean_pearson', 4),
        '',
        '## 2. Training / Validation / Test loss curves',
        '',
        '### Model Comparison',
        '',
        'The plot below shows one row per fold and training, validation, and test loss columns. Each subplot compares image, visual, and multimodal models when histories are available.',
        '',
        f'![Loss curves comparison]({comparison_img_path.name})' if comparison_img_path else '_No checkpoint histories found._',
        '',
        '### Notes',
        '',
        '- Test tables use metrics from each model result folder under each fold.',
        '- Weighted averages are weighted by each model/fold test sample count.',
        '- Missing model results are shown as `N/A` until that fold/model finishes.',
        '- Result folders under `results/old` are excluded.',
        y_axis_note,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate markdown summary and fold-wise loss curves from PathVLM results.')
    results_dir = default_results_dir()
    parser.add_argument('--results-dir', type=Path, default=results_dir, help='Path to results root directory')
    parser.add_argument('--output-md', type=Path, default=results_dir / 'summary.md', help='Output markdown file')
    parser.add_argument('--output-comparison-img', type=Path, default=results_dir / 'loss_comparison.png', help='Output comparison loss plot image')
    parser.add_argument('--output-per-model-img', type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--tissue', type=str, default=None, help='Optional tissue name for summary labels')
    parser.add_argument('--expr-name', type=str, default=None, help='Optional expression target name for summary labels')
    parser.add_argument('--y-min', type=float, default=None, help='Optional lower Y-axis bound for generated loss plots')
    parser.add_argument('--y-max', type=float, default=None, help='Optional upper Y-axis bound for generated loss plots')
    args = parser.parse_args()

    setup_directories()

    if args.y_min is not None and args.y_max is not None and args.y_min >= args.y_max:
        raise ValueError('--y-min must be less than --y-max')

    results_dirs = discover_result_dirs(args.results_dir)
    history_files = discover_history_files(args.results_dir)

    if not results_dirs:
        raise FileNotFoundError(f'No result setup summaries found under {args.results_dir}')

    fold_metrics = collect_fold_metrics(results_dirs)
    if not fold_metrics:
        raise FileNotFoundError(f'No fold/model summaries found under {args.results_dir}')

    comparison_img_path = None
    if history_files:
        comparison_img_path = plot_fold_loss_comparison(
            history_files,
            args.output_comparison_img.parent,
            args.output_comparison_img.name,
            y_min=args.y_min,
            y_max=args.y_max,
        )

    md_path = build_markdown(
        fold_metrics,
        Path(comparison_img_path) if comparison_img_path else None,
        args.output_md,
        args.tissue,
        args.expr_name,
        args.y_min,
        args.y_max,
    )

    print(f'Created summary markdown: {md_path}')
    print(f'Created comparison loss curves: {comparison_img_path}')


if __name__ == '__main__':
    main()
