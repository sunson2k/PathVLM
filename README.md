# PathVLM: Multi-Modal Gene Expression Prediction

A production-ready PyTorch repository for predicting gene expression from spatial transcriptomics data using three complementary input modalities: raw images, visual embeddings, and multimodal embeddings.

## Quick Start

### 1. Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. Data Preparation
Edit `configs/run_config.json` for your local data and output roots:

```json
{
  "data_root": "/path/to/data",
  "project_root": "/path/to/PathVLM",
  "tissue": "Breast",
  "training": {
    "batch_size": 64,
    "num_workers": 4,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "max_epochs": 100,
    "early_stop_patience": 2,
    "device": "cuda",
    "loss_eps": 0.000001
  },
  "model": {
    "resnet_backbone": "resnet50",
    "resnet_pretrained": true,
    "resnet_freeze_backbone": true,
    "dnn_hidden_sizes": [1024, 512],
    "dnn_dropout": 0.4,
    "dnn_normalization": "batchnorm"
  }
}
```

```bash
python scripts/01_prepare_data.py
```
This creates train/val/test splits (70/15/15) with verified alignment across all data modalities.

### 3. Run the End-to-End Pipeline
```bash
python scripts/run_pipeline.py
```
Prepares data, trains three independent models, evaluates each model, and generates final comparison reports:
- **Image Mode**: ResNet50 backbone → shared DNN head → 250-gene output
- **Visual Mode**: 1024-dim embeddings → shared DNN → 250-gene output
- **Multimodal Mode**: 1536-dim (visual+text) embeddings → shared DNN → 250-gene output
- **Reports**: `results/comparison_report.html`, `results/summary.md`, `results/loss_comparison.png`, `results/loss_per_model.png`

Or train individually:
```bash
python scripts/02_train_image.py
python scripts/03_train_visual.py
python scripts/04_train_multimodal.py
```

### 4. Regenerate Reports Only
```bash
python scripts/05_compare_results.py
python scripts/06_summarize_results.py
```

## Data Structure

```
{data_root}/{tissue}/
├── ST-patches/                        # Raw PNG image patches
│   └── {tissue_id}/
│       └── {spot_id}.png
├── ST-features-UNI/                   # Visual embeddings (1024-dim)
│   └── {tissue_id}_features.csv
├── ST-features-CONCH-text/            # Text embeddings (512-dim)
│   └── {tissue_id}_text_features.csv
├── ST-expression-top-8n/              # Expression targets (250 genes)
│   └── {tissue_id}_expression.csv
└── ST-expression-top-spcs/            # Alternative expression targets
    └── {tissue_id}_expression.csv
```

## Project Structure

```
{project_root}/
├── code_plan.md                       # Detailed implementation plan
├── configs/
│   ├── run_config.json                # Local runtime settings
│   └── run_config_template.json       # Example runtime settings
├── src/
│   ├── __init__.py
│   ├── config.py                      # Loads runtime settings from configs/run_config.json
│   ├── data_preparation.py            # Data splitting & alignment verification
│   ├── data_loaders.py                # PyTorch Dataset implementations
│   ├── models.py                      # Three model architectures
│   ├── training.py                    # Training loops & early stopping
│   ├── evaluation.py                  # Metrics computation
│   └── visualization.py               # Loss curves & comparison reports
├── scripts/
│   ├── 01_prepare_data.py             # Data preparation
│   ├── 02_train_image.py              # Train image model
│   ├── 03_train_visual.py             # Train visual model
│   ├── 04_train_multimodal.py         # Train multimodal model
│   ├── 05_compare_results.py          # Generate HTML comparison report
│   ├── 06_summarize_results.py        # Generate markdown summary and loss plots
│   └── run_pipeline.py                # Master orchestrator
├── data_splits/                       # Generated split files
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── results/                           # Model outputs
│   ├── image_mode/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pt
│   │   │   └── history.json
│   │   └── evaluation/
│   │       ├── summary.json
│   │       └── {train|val|test}_*.csv
│   ├── visual_mode/
│   └── multimodal_mode/
└── notebooks/                         # For EDA & visualization
```

## Model Specifications

### ResNetRegressor (Image Mode)
- **Backbone**: ResNet50 (ImageNet pretrained, layers 3-4 trainable)
- **Input**: RGB images (3, 224, 224)
- **Head**: Shared DNN configured by `model.dnn_hidden_sizes`, `model.dnn_dropout`, and `model.dnn_normalization`
- **Architecture**: 2048 → 1024 → 512 → 250 by default
- **Output**: 250 gene expression values

### VisualDNN (Visual Embedding Mode)
- **Input**: 1024-dim visual embeddings (UNI model)
- **Architecture**: 1024 → 1024 → 512 → 250
- **Dropout**: 0.4 between layers
- **Normalization**: BatchNorm inside the DNN by default; StandardScaler fitted on training data only for input features

### MultimodalDNN (Multimodal Mode)
- **Input**: 1536-dim concatenated (1024 visual + 512 text)
- **Architecture**: 1536 → 1024 → 512 → 250
- **Dropout**: 0.4 between layers
- **Normalization**: BatchNorm inside the DNN by default; StandardScaler fitted on training data only for input features

## Training Configuration

```python
batch_size = 64
learning_rate = 3e-4
weight_decay = 1e-4
max_epochs = 100
early_stop_patience = 2
optimizer = Adam
loss_function = scaled_mse_loss  # Per-dimension normalization
```

These values are configured in `configs/run_config.json`.

## Key Features

✅ **Data Integrity**
- Stratified 70/15/15 split grouped by tissue
- Alignment verification across all 5 data modalities
- Gene column consistency validation

✅ **Proper Normalization**
- BatchNorm inside all shared DNN hidden layers by default
- StandardScaler fitted ONLY on training data (prevents leakage)
- Separate scalers per feature mode
- Scaled MSE loss for per-dimension normalization

✅ **Robust Training**
- Early stopping with patience=2 on validation loss
- Learning rate scheduling (ReduceLROnPlateau)
- Best model checkpoint saving by validation performance

✅ **Comprehensive Evaluation**
- Per-gene MSE and Pearson correlation
- Separate train/val/test metrics
- Loss trajectory tracking
- Automatic comparison visualizations

## Evaluation Metrics

Each model generates:
- **Per-Gene Metrics**: MSE, Pearson correlation (r), p-values
- **Split-wise Loss**: Train, validation, test scaled MSE
- **Summaries**: Mean/median gene metrics across all genes
- **Predictions**: CSV with predicted vs. true values per sample

## Output Files

After training completes, results are saved to `{project_root}/results/`:

```
image_mode/
├── checkpoints/
│   ├── best_model.pt              # Best model weights
│   ├── history.json               # Loss trajectory
│   └── checkpoint_epoch_*.pt      # Per-epoch checkpoints
└── evaluation/
    ├── summary.json               # Aggregate metrics
    ├── train_metrics.json         # Per-gene metrics
    ├── val_metrics.json
    ├── test_metrics.json
    ├── train_predictions.csv      # Predictions & targets
    ├── val_predictions.csv
    └── test_predictions.csv

loss_comparison.png               # Loss curves overlay
loss_per_model.png                # Per-model loss curves
comparison_report.html            # HTML summary table
summary.md                        # Markdown comparison summary
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- scikit-learn
- pandas
- numpy
- scipy
- matplotlib
- Pillow

## Troubleshooting

**FileNotFoundError**: Check that data is accessible at `{data_root}/{tissue}/` from `configs/run_config.json`

**CUDA Out of Memory**: Reduce `training.batch_size` in `configs/run_config.json`

**Misaligned Data**: Run `01_prepare_data.py` with debug logging to identify problematic samples

## Implementation Notes

- No data duplication: Split files contain only (tissue_id, spot_id) indices
- Dynamic data loading: Features loaded from disk at sample time
- Memory efficient: Expression CSVs loaded once per dataset creation
- Modular architecture: Each component independently testable
- Configuration centralized: Runtime paths and training hyperparameters in `configs/run_config.json`

## Performance Expectations

On typical Breast tissue data:
- Data prep: ~2-5 minutes
- Image mode: ~30-60 minutes (depends on GPU)
- Visual mode: ~10-20 minutes
- Multimodal mode: ~10-20 minutes
- Evaluation: ~5 minutes

Total pipeline runtime: ~1-2 hours on GPU

## Citation

If you use this pipeline in your research, please reference:
```
PathVLM: Multi-Modal Spatial Transcriptomics Gene Expression Prediction
```
