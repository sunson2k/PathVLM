#!/usr/bin/env python3
"""Generate the PathVLM markdown summary and loss comparison plots."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.results_discovery import default_results_dir, discover_history_files, discover_result_dirs
from src.visualization import plot_loss_curves, plot_per_model_loss_curves


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


def build_markdown(
    table_rows,
    comparison_img_path: Path | None,
    per_model_img_path: Path | None,
    output_path: Path,
    tissue: str | None,
    expr_name: str | None,
):
    title_detail = ''
    if tissue and expr_name:
        title_detail = f' ({tissue} {expr_name})'
    elif tissue:
        title_detail = f' ({tissue})'
    elif expr_name:
        title_detail = f' ({expr_name})'
    metric_prefix = f'{tissue} ' if tissue else ''
    metric_suffix = f' {expr_name}' if expr_name else ''

    setup_names = list(table_rows['mse_loss'].keys())
    header = '| | ' + ' | '.join(setup_names) + ' |'
    separator = '|---|' + '|'.join(['---'] * len(setup_names)) + '|'
    mse_row = (
        f'| {metric_prefix}MSE{metric_suffix} | '
        + ' | '.join(_format_metric(table_rows['mse_loss'].get(name), 6) for name in setup_names)
        + ' |'
    )
    pearson_row = (
        f'| {metric_prefix}Pearson{metric_suffix} | '
        + ' | '.join(_format_metric(table_rows['mean_pearson'].get(name), 4) for name in setup_names)
        + ' |'
    )

    lines = [
        f'# PathVLM Results Summary{title_detail}',
        '',
        f'_Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_',
        '',
        '## 1. Test set final results comparison',
        '',
        header,
        separator,
        mse_row,
        pearson_row,
        '',
        '## 2. Training / Validation / Test loss curves',
        '',
        '### Model Comparison (split-wise view)',
        '',
        'The plot below shows training, validation, and test loss curves for all models on the same plot. All three subplots share the same Y scale.',
        '',
        f'![Loss curves comparison]({comparison_img_path.name})' if comparison_img_path else '_No checkpoint histories found._',
        '',
        '### Per-Model Loss Curves',
        '',
        'The plot below shows each setup separately with its training, validation, and test loss curves. All subplots share the same Y scale for cross-setup comparison.',
        '',
        f'![Loss curves per model]({per_model_img_path.name})' if per_model_img_path else '_No checkpoint histories found._',
        '',
        '### Notes',
        '',
        '- The table uses test split metrics from the best trained model for each discovered setup.',
        '- Result folders under `results/old` are excluded.',
        '- Both loss charts are created from the saved checkpoint histories for each setup.',
        '- All Y-axis scales are synchronized across charts for fair visual comparison.',
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate markdown summary and loss curves from PathVLM results.')
    results_dir = default_results_dir()
    parser.add_argument('--results-dir', type=Path, default=results_dir, help='Path to results root directory')
    parser.add_argument('--output-md', type=Path, default=results_dir / 'summary.md', help='Output markdown file')
    parser.add_argument('--output-comparison-img', type=Path, default=results_dir / 'loss_comparison.png', help='Output comparison loss plot image')
    parser.add_argument('--output-per-model-img', type=Path, default=results_dir / 'loss_per_model.png', help='Output per-model loss plot image')
    parser.add_argument('--tissue', type=str, default=None, help='Optional tissue name for summary labels')
    parser.add_argument('--expr-name', type=str, default=None, help='Optional expression target name for summary labels')
    args = parser.parse_args()

    metrics = {'mse_loss': {}, 'mean_pearson': {}}
    results_dirs = discover_result_dirs(args.results_dir)
    history_files = discover_history_files(args.results_dir)

    if not results_dirs:
        raise FileNotFoundError(f'No result setup summaries found under {args.results_dir}')

    for setup_name, result_dir in results_dirs.items():
        summary = load_summary_metrics(Path(result_dir))
        metrics['mse_loss'][setup_name] = summary['test']['mse_loss']
        metrics['mean_pearson'][setup_name] = summary['test'].get('mean_pearson')

    comparison_img_path = None
    per_model_img_path = None
    if history_files:
        comparison_img_path = plot_loss_curves(history_files, args.output_comparison_img.parent, args.output_comparison_img.name)
        per_model_img_path = plot_per_model_loss_curves(history_files, args.output_per_model_img.parent, args.output_per_model_img.name)
    md_path = build_markdown(
        metrics,
        Path(comparison_img_path) if comparison_img_path else None,
        Path(per_model_img_path) if per_model_img_path else None,
        args.output_md,
        args.tissue,
        args.expr_name,
    )

    print(f'Created summary markdown: {md_path}')
    print(f'Created comparison loss curves: {comparison_img_path}')
    print(f'Created per-model loss curves: {per_model_img_path}')



if __name__ == '__main__':
    main()
