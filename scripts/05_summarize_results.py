#!/usr/bin/env python3
"""Summarize PathVLM results into a markdown report with comparison table and loss curves."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path so we can import visualization
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization import plot_loss_curves, plot_per_model_loss_curves


def load_summary_metrics(results_dir: Path, mode_name: str) -> dict:
    summary_path = results_dir / mode_name / 'evaluation' / 'summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary file: {summary_path}')
    with open(summary_path, 'r') as f:
        return json.load(f)


def build_markdown(table_rows, comparison_img_path: Path, per_model_img_path: Path, output_path: Path, tissue: str, expr_name: str):
    lines = [
        f'# PathVLM Results Summary ({tissue} {expr_name})',
        '',
        f'_Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_',
        '',
        '## 1. Test set final results comparison',
        '',
        '| | Image | Visual | Multimodal |',
        '|---|---|---|---|',
        f'| {tissue} MSE {expr_name} | {table_rows["mse_loss"]["image"]:.6f} | {table_rows["mse_loss"]["visual"]:.6f} | {table_rows["mse_loss"]["multimodal"]:.6f} |',
        f'| {tissue} Pearson {expr_name} | {table_rows["mean_pearson"]["image"]:.4f} | {table_rows["mean_pearson"]["visual"]:.4f} | {table_rows["mean_pearson"]["multimodal"]:.4f} |',
        '',
        '## 2. Training / Validation / Test loss curves',
        '',
        '### Model Comparison (split-wise view)',
        '',
        'The plot below shows training, validation, and test loss curves for all models on the same plot. All three subplots share the same Y scale.',
        '',
        f'![Loss curves comparison]({comparison_img_path.name})',
        '',
        '### Per-Model Loss Curves',
        '',
        'The plot below shows each model separately with its training, validation, and test loss curves. All three subplots share the same Y scale for cross-model comparison.',
        '',
        f'![Loss curves per model]({per_model_img_path.name})',
        '',
        '### Notes',
        '',
        '- The table uses test split metrics from the best trained model for each mode.',
        '- Both loss charts are created from the saved checkpoint histories for each mode.',
        '- All Y-axis scales are synchronized across charts for fair visual comparison.',
    ]
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate markdown summary and loss curves from PathVLM results.')
    parser.add_argument('--results-dir', type=Path, default=Path('results'), help='Path to results root directory')
    parser.add_argument('--output-md', type=Path, default=Path('results/summary.md'), help='Output markdown file')
    parser.add_argument('--output-comparison-img', type=Path, default=Path('results/loss_comparison.png'), help='Output comparison loss plot image')
    parser.add_argument('--output-per-model-img', type=Path, default=Path('results/loss_per_model.png'), help='Output per-model loss plot image')
    parser.add_argument('--tissue', type=str, default='Breast', help='Tissue name for summary labels')
    parser.add_argument('--expr-name', type=str, default='8n', help='Expression target name for summary labels')
    args = parser.parse_args()

    modes = ['image', 'visual', 'multimodal']
    metrics = {'mse_loss': {}, 'mean_pearson': {}}
    history_files = {}

    for mode in modes:
        mode_dir = f'{mode}_mode'
        summary = load_summary_metrics(args.results_dir, mode_dir)
        metrics['mse_loss'][mode] = summary['test']['mse_loss']
        metrics['mean_pearson'][mode] = summary['test']['mean_pearson']
        history_files[mode] = args.results_dir / mode_dir / 'checkpoints' / 'history.json'

    comparison_img_path = plot_loss_curves(history_files, args.output_comparison_img.parent, args.output_comparison_img.name)
    per_model_img_path = plot_per_model_loss_curves(history_files, args.output_per_model_img.parent, args.output_per_model_img.name)
    md_path = build_markdown(metrics, Path(comparison_img_path), Path(per_model_img_path), args.output_md, args.tissue, args.expr_name)

    print(f'Created summary markdown: {md_path}')
    print(f'Created comparison loss curves: {comparison_img_path}')
    print(f'Created per-model loss curves: {per_model_img_path}')



if __name__ == '__main__':
    main()
