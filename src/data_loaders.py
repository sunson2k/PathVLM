"""PyTorch data loaders for three feature modes: image, visual embedding, multimodal."""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

from .config import Config

logger = logging.getLogger(__name__)


class GeneExpressionDataset(Dataset):
    """Base class for gene expression datasets."""
    
    def __init__(self, 
                 split_df: pd.DataFrame,
                 data_root: str,
                 gene_columns: list,
                 expr_csv_dir: str,
                 feature_mode: str = "image",
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            split_df: DataFrame with tissue_id and spot_id columns
            data_root: Root path to data
            gene_columns: List of gene column names
            expr_csv_dir: Directory name for expression CSVs (e.g., ST-expression-top-8n)
            feature_mode: One of ['image', 'visual_embedding', 'multimodal']
            scaler: Optional StandardScaler fitted on training data
        """
        self.split_df = split_df.reset_index(drop=True)
        self.data_root = data_root
        self.gene_columns = gene_columns
        self.expr_csv_dir = expr_csv_dir
        self.feature_mode = feature_mode
        self.scaler = scaler
        
        # Build paths
        self.tissue = Config.data.tissue
        self.patch_root = os.path.join(data_root, self.tissue, Config.data.patch_dir)
        self.visual_feat_root = os.path.join(data_root, self.tissue, Config.data.visual_feat_dir)
        self.text_feat_root = os.path.join(data_root, self.tissue, Config.data.text_feat_dir)
        self.expr_root = os.path.join(data_root, self.tissue, expr_csv_dir)
        
        # Load all expression data into memory for faster access
        self.expr_data = self._load_all_expressions()
        
        logger.info(f"Initialized {feature_mode} dataset with {len(self)} samples")
    
    def _load_all_expressions(self) -> dict:
        """Load all expression CSVs into memory indexed by tissue_id."""
        expr_data = {}
        
        unique_tissues = self.split_df['tissue_id'].unique()
        for tissue_id in unique_tissues:
            expr_path = os.path.join(self.expr_root, f"{tissue_id}_expression.csv")
            df = pd.read_csv(expr_path, index_col=0)
            df.columns = df.columns.str.strip()
            expr_data[tissue_id] = df
        
        return expr_data
    
    def __len__(self) -> int:
        return len(self.split_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            (features, target, uid) where uid is "tissue_id/spot_id"
        """
        row = self.split_df.iloc[idx]
        tissue_id = row['tissue_id']
        spot_id = row['spot_id']
        uid = f"{tissue_id}/{spot_id}"
        
        # Load features (subclasses override this)
        features = self._load_features(tissue_id, spot_id)
        
        # Load target expression
        expr_df = self.expr_data[tissue_id]
        target_values = expr_df.loc[spot_id, self.gene_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        target = torch.from_numpy(target_values)
        
        return features, target, uid
    
    def _load_features(self, tissue_id: str, spot_id: str) -> torch.Tensor:
        """Load features - overridden by subclasses."""
        raise NotImplementedError


class ImageDataset(GeneExpressionDataset):
    """Dataset for raw PNG image patches."""
    
    def __init__(self,
                 split_df: pd.DataFrame,
                 data_root: str,
                 gene_columns: list,
                 expr_csv_dir: str,
                 img_size: int = 224):
        """
        Args:
            img_size: Target image size (224x224 for ResNet)
        """
        super().__init__(split_df, data_root, gene_columns, expr_csv_dir, 
                        feature_mode="image", scaler=None)
        
        self.img_size = img_size
        
        # Standard normalization for ImageNet-pretrained models
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_features(self, tissue_id: str, spot_id: str) -> torch.Tensor:
        """Load and preprocess image."""
        img_path = os.path.join(self.patch_root, tissue_id, f"{spot_id}.png")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        return img_tensor


class VisualEmbeddingDataset(GeneExpressionDataset):
    """Dataset for pre-computed visual embeddings (UNI)."""
    
    def __init__(self,
                 split_df: pd.DataFrame,
                 data_root: str,
                 gene_columns: list,
                 expr_csv_dir: str,
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            scaler: StandardScaler fitted on training visual features
        """
        super().__init__(split_df, data_root, gene_columns, expr_csv_dir,
                        feature_mode="visual_embedding", scaler=scaler)
        
        # Load all visual features into memory
        self.visual_features = self._load_all_visual_features()
    
    def _load_all_visual_features(self) -> dict:
        """Load all visual feature CSVs into memory indexed by tissue_id."""
        visual_features = {}
        
        unique_tissues = self.split_df['tissue_id'].unique()
        for tissue_id in unique_tissues:
            feat_path = os.path.join(self.visual_feat_root, f"{tissue_id}_features.csv")
            df = pd.read_csv(feat_path, index_col='spot_id')
            visual_features[tissue_id] = df
        
        return visual_features
    
    def _load_features(self, tissue_id: str, spot_id: str) -> torch.Tensor:
        """Load visual embedding and optionally apply scaler."""
        feat_df = self.visual_features[tissue_id]
        features = feat_df.loc[spot_id].to_numpy(dtype=np.float32, copy=True)
        
        # Apply scaler if available
        if self.scaler is not None:
            features = self.scaler.transform([features])[0].astype(np.float32, copy=True)
        
        return torch.from_numpy(features)


class MultimodalDataset(GeneExpressionDataset):
    """Dataset for multimodal embeddings (visual + text concatenated)."""
    
    def __init__(self,
                 split_df: pd.DataFrame,
                 data_root: str,
                 gene_columns: list,
                 expr_csv_dir: str,
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            scaler: StandardScaler fitted on training concatenated features
        """
        super().__init__(split_df, data_root, gene_columns, expr_csv_dir,
                        feature_mode="multimodal", scaler=scaler)
        
        # Load all features into memory
        self.visual_features = self._load_all_visual_features()
        self.text_features = self._load_all_text_features()
    
    def _load_all_visual_features(self) -> dict:
        """Load all visual feature CSVs."""
        visual_features = {}
        unique_tissues = self.split_df['tissue_id'].unique()
        for tissue_id in unique_tissues:
            feat_path = os.path.join(self.visual_feat_root, f"{tissue_id}_features.csv")
            df = pd.read_csv(feat_path, index_col='spot_id')
            visual_features[tissue_id] = df
        return visual_features
    
    def _load_all_text_features(self) -> dict:
        """Load all text feature CSVs."""
        text_features = {}
        unique_tissues = self.split_df['tissue_id'].unique()
        for tissue_id in unique_tissues:
            feat_path = os.path.join(self.text_feat_root, f"{tissue_id}_text_features.csv")
            df = pd.read_csv(feat_path, index_col='spot_id')
            text_features[tissue_id] = df
        return text_features
    
    def _load_features(self, tissue_id: str, spot_id: str) -> torch.Tensor:
        """Load and concatenate visual + text embeddings."""
        visual = self.visual_features[tissue_id].loc[spot_id].to_numpy(dtype=np.float32, copy=True)
        text = self.text_features[tissue_id].loc[spot_id].to_numpy(dtype=np.float32, copy=True)
        
        # Concatenate to 1536 dimensions
        combined = np.concatenate([visual, text], axis=0)
        
        # Apply scaler if available
        if self.scaler is not None:
            combined = self.scaler.transform([combined])[0].astype(np.float32, copy=True)
        
        return torch.from_numpy(combined)


def create_dataloaders(split_dir: str,
                      feature_mode: str,
                      batch_size: int = 64,
                      num_workers: int = 4,
                      expr_csv_dir: str = "ST-expression-top-8n",
                      data_root: str = None) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[StandardScaler]]:
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Args:
        split_dir: Directory containing train_split.csv, val_split.csv, test_split.csv
        feature_mode: One of ['image', 'visual_embedding', 'multimodal']
        batch_size: Batch size
        num_workers: Number of workers for data loading
        expr_csv_dir: Expression directory name (ST-expression-top-8n or ST-expression-top-spcs)
        data_root: Root data directory
    
    Returns:
        (train_loader, val_loader, test_loader, scaler)
        scaler is None for image mode, otherwise a fitted StandardScaler
    """
    if data_root is None:
        data_root = Config.data.data_root
    
    # Load split files
    train_df = pd.read_csv(os.path.join(split_dir, 'train_split.csv'))
    val_df = pd.read_csv(os.path.join(split_dir, 'val_split.csv'))
    test_df = pd.read_csv(os.path.join(split_dir, 'test_split.csv'))
    
    # Load gene columns from first expression CSV
    expr_root = os.path.join(data_root, Config.data.tissue, expr_csv_dir)
    first_tissue = train_df.iloc[0]['tissue_id']
    expr_path = os.path.join(expr_root, f"{first_tissue}_expression.csv")
    expr_df = pd.read_csv(expr_path, index_col=0)
    gene_columns = expr_df.columns.str.strip().tolist()
    
    # Initialize scaler if not image mode
    scaler = None
    if feature_mode in ['visual_embedding', 'multimodal']:
        scaler = StandardScaler()
        
        # Fit scaler on training data only
        if feature_mode == 'visual_embedding':
            train_features = []
            for _, row in train_df.iterrows():
                tissue_id = row['tissue_id']
                spot_id = row['spot_id']
                feat_path = os.path.join(data_root, Config.data.tissue, Config.data.visual_feat_dir, 
                                        f"{tissue_id}_features.csv")
                feat_df = pd.read_csv(feat_path, index_col='spot_id')
                train_features.append(feat_df.loc[spot_id].to_numpy(dtype=np.float32, copy=True))
            
            scaler.fit(np.array(train_features))
            logger.info(f"Fitted StandardScaler on {len(train_features)} training samples (visual)")
        
        elif feature_mode == 'multimodal':
            train_features = []
            for _, row in train_df.iterrows():
                tissue_id = row['tissue_id']
                spot_id = row['spot_id']
                
                # Load visual
                visual_path = os.path.join(data_root, Config.data.tissue, Config.data.visual_feat_dir, 
                                          f"{tissue_id}_features.csv")
                visual_df = pd.read_csv(visual_path, index_col='spot_id')
                visual = visual_df.loc[spot_id].to_numpy(dtype=np.float32, copy=True)
                
                # Load text
                text_path = os.path.join(data_root, Config.data.tissue, Config.data.text_feat_dir, 
                                        f"{tissue_id}_text_features.csv")
                text_df = pd.read_csv(text_path, index_col='spot_id')
                text = text_df.loc[spot_id].to_numpy(dtype=np.float32, copy=True)
                
                # Concatenate
                combined = np.concatenate([visual, text], axis=0)
                train_features.append(combined)
            
            scaler.fit(np.array(train_features))
            logger.info(f"Fitted StandardScaler on {len(train_features)} training samples (multimodal)")
    
    # Create datasets
    dataset_class = {
        'image': ImageDataset,
        'visual_embedding': VisualEmbeddingDataset,
        'multimodal': MultimodalDataset
    }[feature_mode]
    
    if feature_mode == 'image':
        train_dataset = dataset_class(train_df, data_root, gene_columns, expr_csv_dir)
        val_dataset = dataset_class(val_df, data_root, gene_columns, expr_csv_dir)
        test_dataset = dataset_class(test_df, data_root, gene_columns, expr_csv_dir)
    else:
        train_dataset = dataset_class(train_df, data_root, gene_columns, expr_csv_dir, scaler=scaler)
        val_dataset = dataset_class(val_df, data_root, gene_columns, expr_csv_dir, scaler=scaler)
        test_dataset = dataset_class(test_df, data_root, gene_columns, expr_csv_dir, scaler=scaler)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    logger.info(f"Created dataloaders for {feature_mode} mode")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, scaler
