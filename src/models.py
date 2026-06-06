"""Model architectures for gene expression prediction."""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class ResNetRegressor(nn.Module):
    """ResNet50-based regressor for image-to-expression prediction."""
    
    def __init__(self, num_genes: int = 250, pretrained: bool = True, freeze_backbone: bool = True):
        """
        Args:
            num_genes: Output dimension (number of genes)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze layers 0-2, train only layers 3-4
        """
        super().__init__()
        
        # Load ResNet50 backbone
        self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Freeze early layers if requested (keep layers 3-4 trainable)
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'layer3' not in name and 'layer4' not in name:
                    param.requires_grad = False
        
        # Replace classification head with regression head
        # ResNet50 outputs 2048 dimensions from layer4
        backbone_out_dim = 2048
        
        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_genes)
        )
        
        # Remove original classification layer
        self.backbone.fc = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, 224, 224)
        
        Returns:
            (batch_size, num_genes)
        """
        x = self.backbone(x)  # (batch_size, 2048)
        x = self.head(x)      # (batch_size, num_genes)
        return x


class MultiLayerDNN(nn.Module):
    """Multi-layer DNN for embedding-based prediction."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int = 250,
                 dropout: float = 0.4):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of genes)
            dropout: Dropout probability between layers
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            (batch_size, output_dim)
        """
        return self.net(x)


class VisualDNN(MultiLayerDNN):
    """DNN for visual embeddings (1024-dim input)."""
    
    def __init__(self, num_genes: int = 250, dropout: float = 0.4):
        """
        Args:
            num_genes: Output dimension
            dropout: Dropout probability
        """
        super().__init__(
            input_dim=1024,
            hidden_dims=[1024, 512],
            output_dim=num_genes,
            dropout=dropout
        )


class MultimodalDNN(MultiLayerDNN):
    """DNN for multimodal embeddings (1536-dim input = 1024 visual + 1024 text)."""
    
    def __init__(self, num_genes: int = 250, dropout: float = 0.4):
        """
        Args:
            num_genes: Output dimension
            dropout: Dropout probability
        """
        super().__init__(
            input_dim=1536,
            hidden_dims=[1024, 512],
            output_dim=num_genes,
            dropout=dropout
        )


def scaled_mse_loss(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Scaled MSE loss: normalize per-dimension to prevent high-magnitude genes from dominating.
    
    Args:
        y_hat: Predicted expression (batch_size, num_genes)
        y: Ground truth expression (batch_size, num_genes)
        eps: Small epsilon for numerical stability
    
    Returns:
        Scalar loss
    """
    # Compute statistics over batch dimension
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    
    y_hat_mean = y_hat.mean(dim=0, keepdim=True)
    y_hat_std = y_hat.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    
    # Normalize
    y_normalized = (y - y_mean) / y_std
    y_hat_normalized = (y_hat - y_hat_mean) / y_hat_std
    
    # MSE on normalized values
    return torch.mean((y_hat_normalized - y_normalized) ** 2)


if __name__ == "__main__":
    # Test model instantiation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test ResNetRegressor
    model1 = ResNetRegressor(num_genes=250, pretrained=True)
    x1 = torch.randn(2, 3, 224, 224)
    y1 = model1(x1)
    assert y1.shape == (2, 250), f"ResNetRegressor output shape mismatch: {y1.shape}"
    print(f"✓ ResNetRegressor output shape: {y1.shape}")
    
    # Test VisualDNN
    model2 = VisualDNN(num_genes=250, dropout=0.4)
    x2 = torch.randn(2, 1024)
    y2 = model2(x2)
    assert y2.shape == (2, 250), f"VisualDNN output shape mismatch: {y2.shape}"
    print(f"✓ VisualDNN output shape: {y2.shape}")
    
    # Test MultimodalDNN
    model3 = MultimodalDNN(num_genes=250, dropout=0.4)
    x3 = torch.randn(2, 1536)
    y3 = model3(x3)
    assert y3.shape == (2, 250), f"MultimodalDNN output shape mismatch: {y3.shape}"
    print(f"✓ MultimodalDNN output shape: {y3.shape}")
    
    # Test loss function
    y_hat = torch.randn(8, 250)
    y = torch.randn(8, 250)
    loss = scaled_mse_loss(y_hat, y)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ scaled_mse_loss: {loss.item():.4f}")
