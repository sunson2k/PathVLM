"""Global configuration for PathVLM pipeline."""
import json
import os
from dataclasses import dataclass


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "run_config.json")
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".cache", "torch"))


def _load_run_config() -> dict:
    """Load user-specific runtime configuration."""
    if not os.path.exists(RUN_CONFIG_PATH):
        return {}

    with open(RUN_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_RUN_CONFIG = _load_run_config()
_TRAINING_CONFIG = _RUN_CONFIG.get("training", {})
_MODEL_CONFIG = _RUN_CONFIG.get("model", {})


def _resolve_from_project_root(path_value: str) -> str:
    """Resolve relative config paths against the repository root."""
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))


@dataclass
class DataConfig:
    """Data configuration."""
    data_root: str = _resolve_from_project_root(_RUN_CONFIG.get("data_root", ""))
    tissue: str = _RUN_CONFIG.get("tissue", "Breast")
    expr_target: str = _RUN_CONFIG.get("expr_target", "8n").lower()
    
    # Data folders
    patch_dir: str = "ST-patches"
    visual_feat_dir: str = "ST-features-UNI"
    text_feat_dir: str = "ST-features-CONCH-text"
    expr_8n_dir: str = "ST-expression-top-8n"
    expr_spcs_dir: str = "ST-expression-top-spcs"
    
    # Dimensions
    num_genes: int = 250
    visual_feature_dim: int = 1024
    text_feature_dim: int = 512
    multimodal_feature_dim: int = 1536  # 1024 + 512
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    @property
    def expr_dir(self) -> str:
        if self.expr_target == "8n":
            return self.expr_8n_dir
        if self.expr_target == "spcs":
            return self.expr_spcs_dir
        raise ValueError(
            f"Unsupported expr_target '{self.expr_target}'. "
            "Use '8n' or 'spcs' in configs/run_config.json."
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = _TRAINING_CONFIG.get("batch_size", 64)
    num_workers: int = _TRAINING_CONFIG.get("num_workers", 4)
    learning_rate: float = _TRAINING_CONFIG.get("learning_rate", 3e-4)
    weight_decay: float = _TRAINING_CONFIG.get("weight_decay", 1e-4)
    max_epochs: int = _TRAINING_CONFIG.get("max_epochs", 100)
    early_stop_patience: int = _TRAINING_CONFIG.get("early_stop_patience", 6)
    device: str = _TRAINING_CONFIG.get("device", "cuda")
    
    # Loss
    loss_eps: float = _TRAINING_CONFIG.get("loss_eps", 1e-6)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # ResNet
    resnet_backbone: str = _MODEL_CONFIG.get("resnet_backbone", "resnet50")
    resnet_pretrained: bool = _MODEL_CONFIG.get("resnet_pretrained", True)
    resnet_freeze_backbone: bool = _MODEL_CONFIG.get("resnet_freeze_backbone", True)
    
    # DNN
    dnn_hidden_sizes: list = None
    dnn_dropout: float = _MODEL_CONFIG.get("dnn_dropout", 0.4)
    
    def __post_init__(self):
        if self.dnn_hidden_sizes is None:
            self.dnn_hidden_sizes = _MODEL_CONFIG.get("dnn_hidden_sizes", [1024, 512])


@dataclass
class PathConfig:
    """Path configuration."""
    project_root: str = _resolve_from_project_root(_RUN_CONFIG.get("project_root", PROJECT_ROOT))
    
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
