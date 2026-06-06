"""Global configuration for PathVLM pipeline."""
import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration."""
    data_root: str = "/scr2/lucasni/data"
    tissue: str = "Breast"
    
    # Data folders
    patch_dir: str = "ST-patches"
    visual_feat_dir: str = "ST-features-UNI"
    text_feat_dir: str = "ST-features-CONCH-text"
    expr_8n_dir: str = "ST-expression-top-8n"
    expr_spcs_dir: str = "ST-expression-top-spcs"
    
    # Dimensions
    num_genes: int = 250
    visual_feature_dim: int = 1024
    text_feature_dim: int = 1024
    multimodal_feature_dim: int = 2048  # 1024 + 1024
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    early_stop_patience: int = 2
    device: str = "cuda"  # will auto-detect if available
    
    # Loss
    loss_eps: float = 1e-6  # epsilon for scaled MSE


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # ResNet
    resnet_backbone: str = "resnet50"
    resnet_pretrained: bool = True
    resnet_freeze_backbone: bool = True
    
    # DNN
    dnn_hidden_sizes: list = None  # Will be set per mode
    dnn_dropout: float = 0.4
    
    def __post_init__(self):
        if self.dnn_hidden_sizes is None:
            self.dnn_hidden_sizes = [1024, 512]


@dataclass
class PathConfig:
    """Path configuration."""
    project_root: str = "/scr2/lucasni/.temp_code/PathVLM"
    
    @property
    def src_dir(self) -> str:
        return os.path.join(self.project_root, "src")
    
    @property
    def scripts_dir(self) -> str:
        return os.path.join(self.project_root, "scripts")
    
    @property
    def data_splits_dir(self) -> str:
        return os.path.join(self.project_root, "data_splits")
    
    @property
    def results_dir(self) -> str:
        return os.path.join(self.project_root, "results")
    
    @property
    def notebooks_dir(self) -> str:
        return os.path.join(self.project_root, "notebooks")


class Config:
    """Master configuration class."""
    data = DataConfig()
    training = TrainingConfig()
    model = ModelConfig()
    paths = PathConfig()


# Create output directories if they don't exist
def setup_directories():
    """Create necessary output directories."""
    os.makedirs(Config.paths.data_splits_dir, exist_ok=True)
    os.makedirs(Config.paths.results_dir, exist_ok=True)
    os.makedirs(Config.paths.notebooks_dir, exist_ok=True)


if __name__ == "__main__":
    setup_directories()
    print("Config setup complete")
