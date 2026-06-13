"""Data preparation: splitting, alignment verification, and consistency checks."""
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreparer:
    """Handles data splitting and alignment verification."""

    def __init__(self, config: Config = Config):
        self.config = config
        self.data_root = config.data.data_root
        self.tissue = config.data.tissue

        self.patch_root = os.path.join(self.data_root, self.tissue, config.data.patch_dir)
        self.visual_feat_root = os.path.join(self.data_root, self.tissue, config.data.visual_feat_dir)
        self.text_feat_root = os.path.join(self.data_root, self.tissue, config.data.text_feat_dir)
        self.expr_8n_root = os.path.join(self.data_root, self.tissue, config.data.expr_8n_dir)
        self.expr_spcs_root = os.path.join(self.data_root, self.tissue, config.data.expr_spcs_dir)

        self.gene_columns = None

    def enumerate_samples(self) -> List[Tuple[str, str]]:
        """Enumerate all patches across tissues."""
        samples = []
        if not os.path.isdir(self.patch_root):
            raise FileNotFoundError(f"Patch root not found: {self.patch_root}")

        tissue_folders = [
            d for d in os.listdir(self.patch_root)
            if os.path.isdir(os.path.join(self.patch_root, d))
        ]
        logger.info(f"Found {len(tissue_folders)} tissue folders")

        for tissue_id in tissue_folders:
            tissue_patch_dir = os.path.join(self.patch_root, tissue_id)
            patches = [
                f.replace('.png', '') for f in os.listdir(tissue_patch_dir)
                if f.endswith('.png')
            ]
            for spot_id in patches:
                samples.append((tissue_id, spot_id))

        logger.info(f"Enumerated {len(samples)} total samples")
        return samples

    def verify_gene_columns(self) -> List[str]:
        """Verify that all expression CSVs share identical column order."""
        logger.info("Verifying gene column consistency across all tissues and expression types...")
        gene_cols_8n = None
        gene_cols_spcs = None

        for expr_file in os.listdir(self.expr_8n_root):
            if expr_file.endswith('_expression.csv'):
                expr_path = os.path.join(self.expr_8n_root, expr_file)
                df = pd.read_csv(expr_path, index_col=0, nrows=0)
                cols = [col.strip() for col in df.columns.tolist()]
                if gene_cols_8n is None:
                    gene_cols_8n = cols
                elif cols != gene_cols_8n:
                    raise ValueError(f"Gene columns mismatch in {expr_file} (8n)")

        for expr_file in os.listdir(self.expr_spcs_root):
            if expr_file.endswith('_expression.csv'):
                expr_path = os.path.join(self.expr_spcs_root, expr_file)
                df = pd.read_csv(expr_path, index_col=0, nrows=0)
                cols = [col.strip() for col in df.columns.tolist()]
                if gene_cols_spcs is None:
                    gene_cols_spcs = cols
                elif cols != gene_cols_spcs:
                    raise ValueError(f"Gene columns mismatch in {expr_file} (spcs)")

        if gene_cols_8n != gene_cols_spcs:
            raise ValueError("Gene columns differ between 8n and spcs expression files")
        if len(gene_cols_8n) != self.config.data.num_genes:
            raise ValueError(f"Expected {self.config.data.num_genes} genes, got {len(gene_cols_8n)}")

        logger.info(f"✓ Gene columns verified: {len(gene_cols_8n)} genes, consistent across all tissues")
        self.gene_columns = gene_cols_8n
        return gene_cols_8n

    def verify_alignment(self, samples: List[Tuple[str, str]]) -> pd.DataFrame:
        """Verify that all samples exist across all modalities."""
        logger.info(f"Verifying alignment of {len(samples)} samples across all modalities...")
        aligned_samples = []

        tissue_ids = sorted({tissue_id for tissue_id, _ in samples})
        for tissue_id in tissue_ids:
            tissue_patch_dir = os.path.join(self.patch_root, tissue_id)
            if not os.path.isdir(tissue_patch_dir):
                logger.warning(f"Patch folder missing for {tissue_id}")
                continue

            patch_ids = {
                f.replace('.png', '') for f in os.listdir(tissue_patch_dir)
                if f.endswith('.png')
            }
            if not patch_ids:
                continue

            visual_feat_path = os.path.join(self.visual_feat_root, f"{tissue_id}_features.csv")
            text_feat_path = os.path.join(self.text_feat_root, f"{tissue_id}_text_features.csv")
            expr_8n_path = os.path.join(self.expr_8n_root, f"{tissue_id}_expression.csv")
            expr_spcs_path = os.path.join(self.expr_spcs_root, f"{tissue_id}_expression.csv")

            if not os.path.exists(visual_feat_path):
                logger.warning(f"Missing visual features for {tissue_id}")
                continue
            if not os.path.exists(text_feat_path):
                logger.warning(f"Missing text features for {tissue_id}")
                continue
            if not os.path.exists(expr_8n_path):
                logger.warning(f"Missing 8n expression for {tissue_id}")
                continue
            if not os.path.exists(expr_spcs_path):
                logger.warning(f"Missing spcs expression for {tissue_id}")
                continue

            visual_df = pd.read_csv(visual_feat_path)
            if 'spot_id' not in visual_df.columns:
                raise ValueError(f"{visual_feat_path} missing 'spot_id' column")
            visual_spots = set(visual_df['spot_id'].astype(str))

            text_df = pd.read_csv(text_feat_path)
            if 'spot_id' not in text_df.columns:
                raise ValueError(f"{text_feat_path} missing 'spot_id' column")
            text_spots = set(text_df['spot_id'].astype(str))

            expr_8n_df = pd.read_csv(expr_8n_path, index_col=0)
            expr_8n_spots = set(expr_8n_df.index.astype(str))

            expr_spcs_df = pd.read_csv(expr_spcs_path, index_col=0)
            expr_spcs_spots = set(expr_spcs_df.index.astype(str))

            valid_spots = patch_ids & visual_spots & text_spots & expr_8n_spots & expr_spcs_spots
            if not valid_spots:
                logger.warning(f"No aligned spots found for tissue {tissue_id}")
                continue

            aligned_samples.extend(
                {'tissue_id': tissue_id, 'spot_id': spot_id}
                for spot_id in sorted(valid_spots)
            )
            logger.info(f"  {tissue_id}: {len(valid_spots)} aligned samples")

        aligned_df = pd.DataFrame(aligned_samples)
        logger.info(f"✓ Alignment verified: {len(aligned_df)} samples aligned across all modalities")
        return aligned_df

    def existing_splits_are_current(self) -> bool:
        """Return True when all configured fold split files and manifest already exist."""
        manifest_path = os.path.join(self.config.paths.data_splits_dir, 'split_manifest.json')
        if self.config.data.force_update_data or not os.path.exists(manifest_path):
            return False

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        expected = {
            'tissue': self.config.data.tissue,
            'split_strategy': self.config.data.split_strategy,
            'num_folds': self.config.data.num_folds,
            'random_seed': self.config.data.random_seed,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return False

        for fold_idx in range(self.config.data.num_folds):
            fold_dir = os.path.join(self.config.paths.data_splits_dir, f"fold_{fold_idx}")
            for split_name in ('train', 'val', 'test'):
                if not os.path.exists(os.path.join(fold_dir, f"{split_name}_split.csv")):
                    return False

        logger.info("✓ Reusing existing data splits: %s", self.config.paths.data_splits_dir)
        logger.info("  Set force_update_data=true to regenerate them.")
        return True

    def apply_cross_validation_splits(self, aligned_df: pd.DataFrame) -> List[Dict]:
        """Create and save configured 6-fold patch-wise or tissue-wise splits."""
        strategy = self.config.data.split_strategy
        if strategy == 'tissue':
            return self.apply_tissue_cross_validation_splits(aligned_df)
        if strategy == 'patch':
            return self.apply_patch_cross_validation_splits(aligned_df)
        raise ValueError("data.split_strategy must be 'patch' or 'tissue'")

    def apply_tissue_cross_validation_splits(self, aligned_df: pd.DataFrame) -> List[Dict]:
        """Create tissue-wise folds with 26/4/6 tissues for 36-tissue 6-fold CV."""
        tissue_ids = np.array(sorted(aligned_df['tissue_id'].unique()))
        num_folds = self.config.data.num_folds
        if num_folds < 2:
            raise ValueError("data.num_folds must be at least 2")
        if len(tissue_ids) < num_folds:
            raise ValueError(f"Cannot create {num_folds} folds from only {len(tissue_ids)} tissues")

        logger.info("Applying %s-fold tissue-wise cross-validation split...", num_folds)
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=self.config.data.random_seed)
        fold_records = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(tissue_ids)):
            remaining_tissues = tissue_ids[train_val_idx].tolist()
            test_tissues = sorted(tissue_ids[test_idx].tolist())
            train_tissues, val_tissues = self._split_train_val_tissues(
                remaining_tissues, fold_idx, total_tissue_count=len(tissue_ids)
            )
            train_df, val_df, test_df = self._select_tissue_splits(
                aligned_df, train_tissues, val_tissues, test_tissues
            )
            fold_records.append(self._save_fold(
                fold_idx, train_df, val_df, test_df,
                train_tissues, val_tissues, test_tissues,
            ))

        self.save_manifest(aligned_df, folds=fold_records)
        return fold_records

    def apply_patch_cross_validation_splits(self, aligned_df: pd.DataFrame) -> List[Dict]:
        """Create patch-wise folds where every aligned spot is tested exactly once."""
        num_folds = self.config.data.num_folds
        if num_folds < 2:
            raise ValueError("data.num_folds must be at least 2")

        logger.info("Applying %s-fold patch-wise cross-validation split...", num_folds)
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=self.config.data.random_seed)
        fold_records = []
        val_fraction_of_train_val = self.config.data.val_ratio / (
            self.config.data.train_ratio + self.config.data.val_ratio
        )

        for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(aligned_df)):
            train_val_df = aligned_df.iloc[train_val_idx].copy()
            test_df = aligned_df.iloc[test_idx].copy()
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_fraction_of_train_val,
                random_state=self.config.data.random_seed + fold_idx,
                stratify=train_val_df['tissue_id'],
            )
            fold_records.append(self._save_fold(
                fold_idx,
                train_df.reset_index(drop=True),
                val_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
                sorted(train_df['tissue_id'].unique().tolist()),
                sorted(val_df['tissue_id'].unique().tolist()),
                sorted(test_df['tissue_id'].unique().tolist()),
            ))

        self.save_manifest(aligned_df, folds=fold_records)
        return fold_records

    def _split_train_val_tissues(
        self,
        tissue_ids: List[str],
        fold_idx: int,
        total_tissue_count: int,
    ) -> Tuple[List[str], List[str]]:
        """Split non-test tissues into train and tissue-held validation groups."""
        if len(tissue_ids) < 2:
            raise ValueError("Each fold needs at least one train and one val tissue")

        rng = np.random.default_rng(self.config.data.random_seed + fold_idx)
        shuffled = np.array(sorted(tissue_ids), dtype=object)
        rng.shuffle(shuffled)

        n_val = max(1, int(round(total_tissue_count * self.config.data.cv_val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        val_tissues = sorted(shuffled[:n_val].tolist())
        train_tissues = sorted(shuffled[n_val:].tolist())
        return train_tissues, val_tissues

    def _select_tissue_splits(
        self,
        aligned_df: pd.DataFrame,
        train_tissues: List[str],
        val_tissues: List[str],
        test_tissues: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Select split rows from tissue ID groups."""
        train_df = aligned_df[aligned_df['tissue_id'].isin(train_tissues)].copy()
        val_df = aligned_df[aligned_df['tissue_id'].isin(val_tissues)].copy()
        test_df = aligned_df[aligned_df['tissue_id'].isin(test_tissues)].copy()
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _save_fold(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_tissues: List[str],
        val_tissues: List[str],
        test_tissues: List[str],
    ) -> Dict:
        fold_name = f"fold_{fold_idx}"
        fold_dir = os.path.join(self.config.paths.data_splits_dir, fold_name)
        self.save_splits(train_df, val_df, test_df, output_dir=fold_dir)
        self._log_tissue_split(fold_name, train_tissues, val_tissues, test_tissues)
        self._log_split_sizes(train_df, val_df, test_df)
        return self._build_split_record(
            fold_name, train_df, val_df, test_df,
            train_tissues, val_tissues, test_tissues,
        )

    def _log_split_sizes(self, train_df, val_df, test_df):
        total = len(train_df) + len(val_df) + len(test_df)
        logger.info("✓ Split complete:")
        logger.info(f"  Train: {len(train_df)} samples ({100*len(train_df)/total:.1f}%)")
        logger.info(f"  Val:   {len(val_df)} samples ({100*len(val_df)/total:.1f}%)")
        logger.info(f"  Test:  {len(test_df)} samples ({100*len(test_df)/total:.1f}%)")

    def _log_tissue_split(
        self,
        split_name: str,
        train_tissues: List[str],
        val_tissues: List[str],
        test_tissues: List[str],
    ):
        logger.info(
            "%s tissues: train=%s, val=%s, test=%s",
            split_name,
            len(train_tissues),
            len(val_tissues),
            len(test_tissues),
        )

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = None,
    ):
        """Save split CSVs to output directory."""
        if output_dir is None:
            output_dir = self.config.paths.data_splits_dir

        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train_split.csv')
        val_path = os.path.join(output_dir, 'val_split.csv')
        test_path = os.path.join(output_dir, 'test_split.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("✓ Splits saved:")
        logger.info(f"  {train_path}")
        logger.info(f"  {val_path}")
        logger.info(f"  {test_path}")

    def save_manifest(self, aligned_df: pd.DataFrame, folds: List[Dict]):
        """Save split metadata for reproducibility and leakage checks."""
        manifest = {
            'tissue': self.config.data.tissue,
            'split_strategy': self.config.data.split_strategy,
            'num_folds': self.config.data.num_folds,
            'cv_enabled_controls_training_only': True,
            'force_update_data': self.config.data.force_update_data,
            'random_seed': self.config.data.random_seed,
            'train_ratio': self.config.data.train_ratio,
            'val_ratio': self.config.data.val_ratio,
            'test_ratio': self.config.data.test_ratio,
            'cv_val_ratio': self.config.data.cv_val_ratio,
            'num_samples': int(len(aligned_df)),
            'num_tissues': int(aligned_df['tissue_id'].nunique()),
            'tissue_sample_counts': {
                tissue_id: int(count)
                for tissue_id, count in aligned_df['tissue_id'].value_counts().sort_index().items()
            },
            'folds': folds,
        }
        os.makedirs(self.config.paths.data_splits_dir, exist_ok=True)
        manifest_path = os.path.join(self.config.paths.data_splits_dir, 'split_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"✓ Split manifest saved: {manifest_path}")

    def _build_split_record(
        self,
        split_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_tissues: List[str],
        val_tissues: List[str],
        test_tissues: List[str],
    ) -> Dict:
        return {
            'name': split_name,
            'train_samples': int(len(train_df)),
            'val_samples': int(len(val_df)),
            'test_samples': int(len(test_df)),
            'train_tissue_count': len(train_tissues),
            'val_tissue_count': len(val_tissues),
            'test_tissue_count': len(test_tissues),
            'train_tissues': train_tissues,
            'val_tissues': val_tissues,
            'test_tissues': test_tissues,
        }

    def load_first_fold(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load fold_0 split files for logging and quick-run compatibility."""
        first_fold_dir = os.path.join(self.config.paths.data_splits_dir, 'fold_0')
        return (
            load_split(os.path.join(first_fold_dir, 'train_split.csv')),
            load_split(os.path.join(first_fold_dir, 'val_split.csv')),
            load_split(os.path.join(first_fold_dir, 'test_split.csv')),
        )

    def run_full_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute full data preparation pipeline.

        Returns:
            Tuple of fold_0 (train_df, val_df, test_df) for logging compatibility.
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PREPARATION PIPELINE")
        logger.info("=" * 60)
        logger.info("Split output directory: %s", self.config.paths.data_splits_dir)

        if self.existing_splits_are_current():
            train_df, val_df, test_df = self.load_first_fold()
        else:
            self.verify_gene_columns()
            samples = self.enumerate_samples()
            aligned_df = self.verify_alignment(samples)
            self.apply_cross_validation_splits(aligned_df)
            train_df, val_df, test_df = self.load_first_fold()

        logger.info("=" * 60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 60)
        return train_df, val_df, test_df


def load_split(split_path: str) -> pd.DataFrame:
    """Load a split CSV file."""
    return pd.read_csv(split_path)
