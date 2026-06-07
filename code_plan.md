# PathVLM: Production-Ready Multi-Modal Gene Expression Prediction
## Complete Implementation Plan

---

## Executive Summary
Build a modularized, clean PyTorch-based repository for predicting gene expression from spatial transcriptomics data using three input modalities: (1) raw images, (2) visual embeddings, and (3) multimodal (visual + text) embeddings. All three models will be trained on the same consistent 70/15/15 train/val/test split and evaluated comparatively.

---

## Part 1: Data Splitting & Indexing Strategy

### 1.1 Data Structure Overview
**Root:** `{data_root}/{tissue}/` from `scripts/run_config.json`

| Component | Format | Structure | Dimensions |
|-----------|--------|-----------|-----------|
| Raw Images | PNG files | `ST-patches/{tissue_id}/{spot_id}.png` | Variable spatial dims |
| Visual Features (UNI) | CSV | `ST-features-UNI/{tissue_id}_features.csv` | Nspots × 1024 |
| Text Features (CONCH) | CSV | `ST-features-CONCH-text/{tissue_id}_text_features.csv` | Nspots × 1024 |
| Expression (8n) | CSV | `ST-expression-top-8n/{tissue_id}_expression.csv` | Nspots × 250 |
| Expression (spcs) | CSV | `ST-expression-top-spcs/{tissue_id}_expression.csv` | Nspots × 250 |

**Key Constraint:** Each CSV has a `spot_id` index and tissue ID is derived from filename.

### 1.2 Splitting Logic

**Goal:** Create a consistent 70/15/15 split across ALL tissues and ALL modalities.

**Process:**
1. **Enumerate all samples** across tissues:
   - For each `{tissue_id}` folder in `ST-patches/`
   - List all `{spot_id}.png` files
   - Create unique identifier: `{tissue_id}_{spot_id}` (or `{tissue_id}/{spot_id}`)
   - Collect all spot pairs (tissue, spot)

2. **Verify alignment** (critical step):
   - For each (tissue, spot) pair, verify existence in:
     - `ST-features-UNI/{tissue_id}_features.csv` (row with matching spot_id)
     - `ST-features-CONCH-text/{tissue_id}_text_features.csv`
     - `ST-expression-top-8n/{tissue_id}_expression.csv`
     - `ST-expression-top-spcs/{tissue_id}_expression.csv`
   - Skip any sample missing from ANY modality

3. **Verify gene column consistency:**
   - Load gene columns from one sample of each expression CSV
   - Assert all tissues share identical column order (genes) in both 8n and spcs
   - If not, raise an error and halt

4. **Apply stratified split:**
   - Group samples by tissue to maintain tissue balance
   - Use `sklearn.model_selection.train_test_split` with stratification on tissue
   - Create: train (70%), temp (30%)
   - From temp: val (50% = 15%), test (50% = 15%)

5. **Save split files:**
   - Output directory: `{project_root}/data_splits/`
   - Three CSV files: `train_split.csv`, `val_split.csv`, `test_split.csv`
   - Each CSV contains columns: `tissue_id`, `spot_id` (no data duplication)

### 1.3 Implementation Module: `data_preparation.py`

**Location:** `{project_root}/src/data_preparation.py`

**Key Functions:**
- `enumerate_samples()` → List[(tissue_id, spot_id)]
- `verify_alignment()` → None (raises on mismatch)
- `verify_gene_columns()` → None (raises on inconsistency)
- `apply_stratified_split()` → (train_df, val_df, test_df)
- `save_splits()` → None (writes CSVs)
- `load_split(split_path)` → DataFrame

---

## Part 2: Modular Data Loaders

### 2.1 Core Data Classes

**Module:** `{project_root}/src/data_loaders.py`

#### 2.1.1 GeneExpressionDataset (Abstract Base)

```
class GeneExpressionDataset(Dataset):
    - __init__(split_df, feature_mode, data_root, scaler=None)
    - __len__() → int
    - __getitem__(idx) → (features, target, metadata)
```

**Parameters:**
- `feature_mode`: one of ['image', 'visual_embedding', 'multimodal']
- `scaler`: StandardScaler fitted on training data (None for images)

#### 2.1.2 Concrete Implementations

**Mode 1: ImageDataset(GeneExpressionDataset)**
- Load raw PNG from `ST-patches/{tissue}/{spot_id}.png`
- Apply torchvision transforms (resize to 224×224, normalize)
- Load 250 target genes from expression CSV
- Return: (image_tensor, target_vector, uid)

**Mode 2: VisualEmbeddingDataset(GeneExpressionDataset)**
- Load pre-computed features from `ST-features-UNI/{tissue}_features.csv`
- Load same target
- Apply StandardScaler (fitted on training split only)
- Return: (scaled_feature_1024, target_vector, uid)

**Mode 3: MultimodalDataset(GeneExpressionDataset)**
- Load visual + text embeddings (two 1024-dim vectors)
- Concatenate into 1536-dim vector
- Apply StandardScaler to concatenated features
- Return: (scaled_multimodal_1536, target_vector, uid)

### 2.2 DataLoader Factory

**Function:** `create_dataloaders(split_dir, feature_mode, batch_size, num_workers)`
- Loads train/val/test splits from CSV
- Fits StandardScaler on training features (if applicable)
- Creates train/val/test DataLoaders with proper shuffle settings
- Returns: (train_loader, val_loader, test_loader, scaler)

---

## Part 3: Model Architecture & Training

### 3.1 Model Implementations

**Module:** `{project_root}/src/models.py`

#### Mode 1: ResNetRegressor

**Base:** Adapt existing `GeneCNN` implementation if available

```
class ResNetRegressor(nn.Module):
    - backbone: ResNet50 (pretrained, layer3+layer4 unfrozen)
    - head: 2048 → 1024 → 512 → 250
    - forward(x) → (batch, 250)
```

#### Mode 2: MultiLayerDNN (Visual Embeddings)

**Base:** Adapt existing `GeneMLP` implementation if available

```
class MultiLayerDNN(nn.Module):
    - input_dim: 1024
    - layers: 1024 → 1024 → 512 → 250
    - dropout: 0.4 between layers
    - forward(x) → (batch, 250)
```

#### Mode 3: MultimodalDNN

```
class MultimodalDNN(nn.Module):
    - input_dim: 1536 (1024 visual + 1024 text)
    - layers: 1536 → 1024 → 512 → 250
    - dropout: 0.4
    - forward(x) → (batch, 250)
```

### 3.2 Loss Function

**Reference:** Existing `scaled_mse_loss` from codebase

```python
def scaled_mse_loss(y_hat, y, eps=1e-6):
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    p_mean = y_hat.mean(dim=0, keepdim=True)
    p_std = y_hat.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    y_z = (y - y_mean) / y_std
    p_z = (y_hat - p_mean) / p_std
    return torch.mean((p_z - y_z) ** 2)
```

### 3.3 Training Module

**Module:** `{project_root}/src/training.py`

**Configuration (CFG):**
```
- batch_size: 64
- num_workers: 4
- learning_rate: 3e-4
- weight_decay: 1e-4
- epochs: 100 (early stop at patience=2)
- device: auto-detect CUDA
```

Training and model hyperparameters are loaded from `scripts/run_config.json`.

---

## Part 4: Evaluation & Reporting

**Module:** `{project_root}/src/evaluation.py`

**Tracked Metrics (per epoch):**
- Train loss (scaled MSE)
- Validation loss
- Test loss

---

## Part 5: Repository Structure

```
{project_root}/
├── code_plan.md (THIS FILE)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preparation.py
│   ├── data_loaders.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_image.py
│   ├── 03_train_visual.py
│   ├── 04_train_multimodal.py
│   ├── 05_evaluate_all.py
│   └── run_pipeline.py
├── data_splits/
├── results/
└── notebooks/
```

---

## Key Implementation Details

### Constraints

1. **StandardScaler Fitting:** ONLY fit on training data
2. **Gene Column Order:** Verify identical order across all tissue expression CSVs
3. **Early Stopping:** patience=2 on validation loss
4. **Loss Normalization:** Per-dimension scaling via scaled_mse_loss
5. **Hyperparameters:** shared across modes and configured in `scripts/run_config.json`

---

## Execution Roadmap

### Phase 1: Data Preparation
`01_prepare_data.py` → Create and save train/val/test splits

### Phase 2: Model Training (can parallelize)
- `02_train_image.py` → Train ResNet
- `03_train_visual.py` → Train visual DNN
- `04_train_multimodal.py` → Train multimodal DNN

### Phase 3: Evaluation
`05_evaluate_all.py` → Aggregate results and generate comparison report

### Phase 4: Master Orchestrator
`run_pipeline.py` → Run all phases in sequence
