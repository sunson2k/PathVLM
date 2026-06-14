"""Global configuration for PathVLM pipeline."""
import json
import os
import shutil
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
_DATA_CONFIG = _RUN_CONFIG.get("data", {})
_TRAINING_CONFIG = _RUN_CONFIG.get("training", {})
_MODEL_CONFIG = _RUN_CONFIG.get("model", {})


def _resolve_from_project_root(path_value: str) -> str:
    """Resolve relative config paths against the repository root."""
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))


def _safe_path_part(value: str) -> str:
    """Return a filesystem-safe path segment for experiment folder names."""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


@dataclass
class DataConfig:
    """Data configuration."""
    data_root: str = _resolve_from_project_root(_RUN_CONFIG.get("data_root", ""))
    tissue: str = _DATA_CONFIG.get("tissue", _RUN_CONFIG.get("tissue", "Breast"))
    expr_target: str = _DATA_CONFIG.get(
        "expr_target",
        _RUN_CONFIG.get("expr_target", "8n"),
    ).lower()
    
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
    train_ratio: float = _DATA_CONFIG.get("train_ratio", _RUN_CONFIG.get("train_ratio", 0.75))
    val_ratio: float = _DATA_CONFIG.get("val_ratio", _RUN_CONFIG.get("val_ratio", 0.10))
    test_ratio: float = _DATA_CONFIG.get("test_ratio", _RUN_CONFIG.get("test_ratio", 0.15))
    random_seed: int = _DATA_CONFIG.get("random_seed", _RUN_CONFIG.get("random_seed", 42))
    split_strategy: str = _DATA_CONFIG.get(
        "split_strategy",
        _RUN_CONFIG.get("split_strategy", "patch"),
    ).lower()
    cv_enabled: bool = _DATA_CONFIG.get("cv_enabled", _RUN_CONFIG.get("cv_enabled", False))
    num_folds: int = _DATA_CONFIG.get("num_folds", _RUN_CONFIG.get("num_folds", 6))
    cv_val_ratio: float = _DATA_CONFIG.get("cv_val_ratio", _RUN_CONFIG.get("cv_val_ratio", 0.10))
    force_update_data: bool = _DATA_CONFIG.get(
        "force_update_data",
        _RUN_CONFIG.get("force_update_data", False),
    )

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
    device: str = os.environ.get(
        "PATHVLM_TRAINING_DEVICE",
        _TRAINING_CONFIG.get("device", "cuda"),
    )
    
    # Loss
    loss_eps: float = _TRAINING_CONFIG.get("loss_eps", 1e-6)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # ResNet
    resnet_backbone: str = _MODEL_CONFIG.get("resnet_backbone", "resnet50")
    resnet_source: str = _MODEL_CONFIG.get("resnet_source", "torchvision")
    resnet_local_path: str = _MODEL_CONFIG.get(
        "resnet_local_path",
        "local_models/microsoft_resnet-50",
    )
    resnet_pretrained: bool = _MODEL_CONFIG.get("resnet_pretrained", True)
    resnet_freeze_mode: str = _MODEL_CONFIG.get("resnet_freeze_mode", "early")
    
    # DNN
    dnn_hidden_sizes: list = None
    dnn_dropout: float = _MODEL_CONFIG.get("dnn_dropout", 0.4)
    dnn_normalization: str = _MODEL_CONFIG.get("dnn_normalization", "batchnorm")
    multimodal_model: str = _MODEL_CONFIG.get("multimodal_model", "cross_attention")
    
    def __post_init__(self):
        self.resnet_source = self.resnet_source.lower()
        self.resnet_local_path = _resolve_from_project_root(self.resnet_local_path)
        self.resnet_freeze_mode = self.resnet_freeze_mode.lower()
        self.dnn_normalization = self.dnn_normalization.lower()
        self.multimodal_model = self.multimodal_model.lower()
        if self.dnn_hidden_sizes is None:
            self.dnn_hidden_sizes = _MODEL_CONFIG.get("dnn_hidden_sizes", [1024, 512])


@dataclass
class PathConfig:
    """Path configuration."""
    project_root: str = _resolve_from_project_root(_RUN_CONFIG.get("project_root", PROJECT_ROOT))

    @property
    def experiment_name(self) -> str:
        tissue = _safe_path_part(Config.data.tissue)
        expr_target = _safe_path_part(Config.data.expr_target)
        strategy = _safe_path_part(Config.data.split_strategy)
        return f"exp_{tissue}_{expr_target}_{strategy}"

    @property
    def experiment_results_root(self) -> str:
        return os.path.join(self.project_root, "experiment_results")

    @property
    def experiment_dir(self) -> str:
        return os.path.join(self.experiment_results_root, self.experiment_name)
    
    @property
    def src_dir(self) -> str:
        return os.path.join(self.project_root, "src")
    
    @property
    def scripts_dir(self) -> str:
        return os.path.join(self.project_root, "scripts")
    
    @property
    def data_splits_dir(self) -> str:
        tissue = _safe_path_part(Config.data.tissue)
        strategy = _safe_path_part(Config.data.split_strategy)
        return os.path.join(
            self.project_root,
            "data_splits",
            f"{tissue}_{strategy}_{Config.data.num_folds}fold",
        )
    
    @property
    def results_dir(self) -> str:
        return os.path.join(self.experiment_dir, "results")

    @property
    def image_results_name(self) -> str:
        if Config.model.resnet_source == "torchvision":
            return f"image_{Config.model.resnet_freeze_mode}"
        return f"image_{Config.model.resnet_source}_{Config.model.resnet_freeze_mode}"

    @property
    def visual_results_name(self) -> str:
        return "visual_mode"

    @property
    def multimodal_results_name(self) -> str:
        return f"multimodal_{Config.model.multimodal_model}"

    @property
    def image_results_dir(self) -> str:
        return os.path.join(self.results_dir, self.image_results_name)

    @property
    def visual_results_dir(self) -> str:
        return os.path.join(self.results_dir, self.visual_results_name)

    @property
    def multimodal_results_dir(self) -> str:
        return os.path.join(self.results_dir, self.multimodal_results_name)

    @property
    def model_results_dirs(self) -> dict:
        return {
            "image": self.image_results_dir,
            "visual": self.visual_results_dir,
            "multimodal": self.multimodal_results_dir,
        }
    
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
    os.makedirs(Config.paths.experiment_results_root, exist_ok=True)
    os.makedirs(Config.paths.experiment_dir, exist_ok=True)
    os.makedirs(Config.paths.data_splits_dir, exist_ok=True)
    os.makedirs(Config.paths.results_dir, exist_ok=True)
    os.makedirs(Config.paths.notebooks_dir, exist_ok=True)
    snapshot_run_config()


def snapshot_run_config():
    """Copy the active run config into the experiment folder."""
    if not os.path.exists(RUN_CONFIG_PATH):
        return

    os.makedirs(Config.paths.experiment_dir, exist_ok=True)
    snapshot_path = os.path.join(Config.paths.experiment_dir, "run_config.json")
    shutil.copy2(RUN_CONFIG_PATH, snapshot_path)


if __name__ == "__main__":
    setup_directories()
    print("Config setup complete")
