"""Data preparation: splitting, alignment verification, and consistency checks."""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import logging

from .config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreparer:
    """Handles data splitting and alignment verification."""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.data_root = config.data.data_root
        self.tissue = config.data.tissue
        
        # Build full paths
        self.patch_root = os.path.join(self.data_root, self.tissue, config.data.patch_dir)
        self.visual_feat_root = os.path.join(self.data_root, self.tissue, config.data.visual_feat_dir)
        self.text_feat_root = os.path.join(self.data_root, self.tissue, config.data.text_feat_dir)
        self.expr_8n_root = os.path.join(self.data_root, self.tissue, config.data.expr_8n_dir)
        self.expr_spcs_root = os.path.join(self.data_root, self.tissue, config.data.expr_spcs_dir)
        
        self.gene_columns = None  # Will be loaded during verification
    
    def enumerate_samples(self) -> List[Tuple[str, str]]:
        """
        Enumerate all patches across tissues.
        
        Returns:
            List of (tissue_id, spot_id) tuples
        """
        samples = []
        
        # List all tissue folders in ST-patches
        if not os.path.isdir(self.patch_root):
            raise FileNotFoundError(f"Patch root not found: {self.patch_root}")
        
        tissue_folders = [d for d in os.listdir(self.patch_root) 
                         if os.path.isdir(os.path.join(self.patch_root, d))]
        
        logger.info(f"Found {len(tissue_folders)} tissue folders")
        
        for tissue_id in tissue_folders:
            tissue_patch_dir = os.path.join(self.patch_root, tissue_id)
            patches = [f.replace('.png', '') for f in os.listdir(tissue_patch_dir) 
                      if f.endswith('.png')]
            
            for spot_id in patches:
                samples.append((tissue_id, spot_id))
        
        logger.info(f"Enumerated {len(samples)} total samples")
        return samples
    
    def verify_gene_columns(self) -> List[str]:
        """
        Verify that all expression CSVs share identical column order.
        
        Returns:
            List of gene column names
        
        Raises:
            ValueError if gene columns don't match across tissues or files
        """
        logger.info("Verifying gene column consistency across all tissues and expression types...")
        
        # Load gene columns from 8n and spcs
        gene_cols_8n = None
        gene_cols_spcs = None
        
        for expr_file in os.listdir(self.expr_8n_root):
            if expr_file.endswith('_expression.csv'):
                expr_path = os.path.join(self.expr_8n_root, expr_file)
                df = pd.read_csv(expr_path, index_col=0, nrows=0)  # Load only gene column names
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
        
        # Check if 8n and spcs have same columns
        if gene_cols_8n != gene_cols_spcs:
            raise ValueError("Gene columns differ between 8n and spcs expression files")
        
        if len(gene_cols_8n) != self.config.data.num_genes:
            raise ValueError(f"Expected {self.config.data.num_genes} genes, got {len(gene_cols_8n)}")
        
        logger.info(f"✓ Gene columns verified: {len(gene_cols_8n)} genes, consistent across all tissues")
        self.gene_columns = gene_cols_8n
        return gene_cols_8n
    
    def verify_alignment(self, samples: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Verify that all samples exist across all modalities.
        
        Args:
            samples: List of (tissue_id, spot_id) tuples
        
        Returns:
            DataFrame with aligned samples (tissue_id, spot_id columns)
        
        Raises:
            ValueError if modalities are misaligned
        """
        logger.info(f"Verifying alignment of {len(samples)} samples across all modalities...")
        
        aligned_samples = []
        
        # Create a lookup for unique tissue folders and avoid reloading modality files repeatedly.
        tissue_ids = sorted({tissue_id for tissue_id, _ in samples})
        for tissue_id in tissue_ids:
            tissue_patch_dir = os.path.join(self.patch_root, tissue_id)
            if not os.path.isdir(tissue_patch_dir):
                logger.warning(f"Patch folder missing for {tissue_id}")
                continue
            
            patch_ids = {f.replace('.png', '') for f in os.listdir(tissue_patch_dir) if f.endswith('.png')}
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
            
            aligned_samples.extend({'tissue_id': tissue_id, 'spot_id': spot_id}
                                   for spot_id in sorted(valid_spots))
            
            logger.info(f"  {tissue_id}: {len(valid_spots)} aligned samples")
        
        aligned_df = pd.DataFrame(aligned_samples)
        logger.info(f"✓ Alignment verified: {len(aligned_df)} samples aligned across all modalities")
        
        return aligned_df
    
    def apply_stratified_split(self, 
                               aligned_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply stratified 70/15/15 split grouped by tissue.
        
        Args:
            aligned_df: DataFrame with (tissue_id, spot_id) columns
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Applying stratified 70/15/15 split...")
        
        # Set random seed for reproducibility
        np.random.seed(self.config.data.random_seed)
        
        # Train / temp split (70/30)
        train_df, temp_df = train_test_split(
            aligned_df,
            test_size=0.3,
            random_state=self.config.data.random_seed,
            stratify=aligned_df['tissue_id']
        )
        
        # Val / test split (50/50 of temp = 15% each)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=self.config.data.random_seed,
            stratify=temp_df['tissue_id']
        )
        
        logger.info(f"✓ Split complete:")
        logger.info(f"  Train: {len(train_df)} samples ({100*len(train_df)/len(aligned_df):.1f}%)")
        logger.info(f"  Val:   {len(val_df)} samples ({100*len(val_df)/len(aligned_df):.1f}%)")
        logger.info(f"  Test:  {len(test_df)} samples ({100*len(test_df)/len(aligned_df):.1f}%)")
        
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def save_splits(self, 
                   train_df: pd.DataFrame, 
                   val_df: pd.DataFrame, 
                   test_df: pd.DataFrame,
                   output_dir: str = None):
        """
        Save split CSVs to output directory.
        
        Args:
            train_df: Training split DataFrame
            val_df: Validation split DataFrame
            test_df: Test split DataFrame
            output_dir: Output directory (defaults to data_splits_dir)
        """
        if output_dir is None:
            output_dir = self.config.paths.data_splits_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, 'train_split.csv')
        val_path = os.path.join(output_dir, 'val_split.csv')
        test_path = os.path.join(output_dir, 'test_split.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✓ Splits saved:")
        logger.info(f"  {train_path}")
        logger.info(f"  {val_path}")
        logger.info(f"  {test_path}")
    
    def run_full_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute full data preparation pipeline.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PREPARATION PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Verify gene columns
        self.verify_gene_columns()
        
        # Step 2: Enumerate all samples
        samples = self.enumerate_samples()
        
        # Step 3: Verify alignment
        aligned_df = self.verify_alignment(samples)
        
        # Step 4: Apply split
        train_df, val_df, test_df = self.apply_stratified_split(aligned_df)
        
        # Step 5: Save splits
        self.save_splits(train_df, val_df, test_df)
        
        logger.info("=" * 60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 60)
        
        return train_df, val_df, test_df


def load_split(split_path: str) -> pd.DataFrame:
    """Load a split CSV file."""
    return pd.read_csv(split_path)
