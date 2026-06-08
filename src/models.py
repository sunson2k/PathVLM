"""Model architectures for gene expression prediction."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class ResNetRegressor(nn.Module):
    """ResNet50-based regressor for image-to-expression prediction."""

    def __init__(
        self,
        num_genes: int = 250,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_mode: str = "early",
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
        normalization: str = "batchnorm",
    ):
        """
        Args:
            num_genes: Output dimension (number of genes)
            backbone: ResNet backbone name
            pretrained: Use ImageNet pretrained weights
            freeze_mode: ResNet training mode. One of "none", "early", or "all".
                "early" freezes all layers except layer3/layer4. "all" freezes
                the full ResNet backbone.
            hidden_dims: Regression head hidden layer dimensions
            dropout: Dropout probability between head layers
            normalization: Normalization layer for head hidden layers
        """
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")

        # Load ResNet50 backbone
        self.backbone = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)

        freeze_mode = freeze_mode.lower()
        if freeze_mode not in {"none", "early", "all"}:
            raise ValueError(
                "Unsupported ResNet freeze mode "
                f"'{freeze_mode}'. Use 'none', 'early', or 'all'."
            )

        if freeze_mode == "all":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_mode == "early":
            for name, param in self.backbone.named_parameters():
                if "layer3" not in name and "layer4" not in name:
                    param.requires_grad = False

        # Replace classification head with regression head
        # ResNet50 outputs 2048 dimensions from layer4
        backbone_out_dim = 2048

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        self.head = MultiLayerDNN(
            input_dim=backbone_out_dim,
            hidden_dims=hidden_dims,
            output_dim=num_genes,
            dropout=dropout,
            normalization=normalization,
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
        x = self.head(x)  # (batch_size, num_genes)
        return x


class MultiLayerDNN(nn.Module):
    """Configurable MLP regressor used by all prediction heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 250,
        dropout: float = 0.4,
        normalization: str = "batchnorm",
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of genes)
            dropout: Dropout probability between layers
            normalization: One of "batchnorm", "layernorm", or "none"
        """
        super().__init__()
        normalization = normalization.lower()
        if normalization not in {"batchnorm", "layernorm", "none"}:
            raise ValueError(
                "Unsupported DNN normalization "
                f"'{normalization}'. Use 'batchnorm', 'layernorm', or 'none'."
            )

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
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

    def __init__(
        self,
        num_genes: int = 250,
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
        normalization: str = "batchnorm",
    ):
        """
        Args:
            num_genes: Output dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            normalization: Normalization layer for hidden layers
        """
        if hidden_dims is None:
            hidden_dims = [1024, 512]

        super().__init__(
            input_dim=1024,
            hidden_dims=hidden_dims,
            output_dim=num_genes,
            dropout=dropout,
            normalization=normalization,
        )


class MultimodalDNN(MultiLayerDNN):
    """DNN for multimodal embeddings (1536-dim input = 1024 visual + 512 text)."""

    def __init__(
        self,
        num_genes: int = 250,
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
        normalization: str = "batchnorm",
    ):
        """
        Args:
            num_genes: Output dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            normalization: Normalization layer for hidden layers
        """
        if hidden_dims is None:
            hidden_dims = [1024, 512]

        super().__init__(
            input_dim=1536,
            hidden_dims=hidden_dims,
            output_dim=num_genes,
            dropout=dropout,
            normalization=normalization,
        )


class MultimodalCrossAttentionDNN(nn.Module):
    """DNN that fuses UNI (1024) and CONCH (512) embeddings using Cross-Attention

    before predicting gene expression.
    """

    def __init__(
        self,
        num_genes: int = 250,
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
        embed_dim: int = 512,
        num_heads: int = 8,
        visual_input_dim: int = 1024,
        text_input_dim: int = 512,
        normalization: str = "batchnorm",
    ):
        """
        Args:
            num_genes: Output dimension (number of genes)
            hidden_dims: Downstream MLP hidden layer dimensions
            dropout: Dropout probability
            embed_dim: The joint projection space size for attention
            num_heads: Number of attention heads (must cleanly divide embed_dim)
            visual_input_dim: Input dimension for UNI visual embeddings
            text_input_dim: Input dimension for CONCH text embeddings
            normalization: Normalization layer for downstream MLP hidden layers
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]  # Tailored for the 512-dim post-attention input

        self.visual_input_dim = visual_input_dim
        self.text_input_dim = text_input_dim

        # 1. Linear projections to align modalities into the same dimension
        self.visual_projection = nn.Linear(visual_input_dim, embed_dim)
        self.text_projection = nn.Linear(text_input_dim, embed_dim)

        # 2. Cross-Attention Layer
        # batch_first=True expects tensors of shape (batch, sequence_len, embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer Norm to stabilize the output of attention before the MLP
        self.ln = nn.LayerNorm(embed_dim)

        # 3. Downstream Regressor Head
        self.regressor = MultiLayerDNN(
            input_dim=embed_dim,  # Now takes the 512-dim attended embedding
            hidden_dims=hidden_dims,
            output_dim=num_genes,
            dropout=dropout,
            normalization=normalization,
        )

    def forward(
        self, visual_x: torch.Tensor, text_x: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            visual_x: Either UNI embeddings, shape (batch_size, 1024), or
                concatenated UNI + CONCH embeddings, shape (batch_size, 1536)
            text_x: Optional CONCH embeddings, shape (batch_size, 512)
        Returns:
            Continuous predictions, shape (batch_size, num_genes)
        """
        if text_x is None:
            expected_dim = self.visual_input_dim + self.text_input_dim
            if visual_x.dim() != 2 or visual_x.size(1) != expected_dim:
                raise ValueError(
                    "MultimodalCrossAttentionDNN expected either "
                    f"(visual_x, text_x) or one concatenated tensor with "
                    f"{expected_dim} features, got shape {tuple(visual_x.shape)}."
                )
            visual_x, text_x = torch.split(
                visual_x, [self.visual_input_dim, self.text_input_dim], dim=1
            )

        # Step 1: Project to common embedding size
        # (batch_size, embed_dim)
        v_proj = self.visual_projection(visual_x)
        t_proj = self.text_projection(text_x)

        # Step 2: Add placeholder sequence dimension for nn.MultiheadAttention
        # (batch_size, 1, embed_dim)
        v_seq = v_proj.unsqueeze(1)
        t_seq = t_proj.unsqueeze(1)

        # Step 3: Cross Attention
        # Query = Visual (UNI), Key & Value = Text (CONCH)
        # This forces text features to modulate/contextualize visual features
        attn_output, _ = self.cross_attention(query=v_seq, key=t_seq, value=t_seq)

        # 4. THE RESIDUAL CONNECTION (Crucial for preserving visual quality)
        # We add the original visual Query to the attention output
        fused_seq = v_seq + attn_output

        # 5. LayerNorm and Squeeze
        fused_embedding = self.ln(fused_seq.squeeze(1))

        # 6. Downstream Regression
        return self.regressor(fused_embedding)


class MultimodalGMUDNN(nn.Module):
    """Placeholder for a multimodal Gated Multimodal Unit model."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "MultimodalGMUDNN is a placeholder. Implement the GMU architecture "
            "in src/models.py before selecting model.multimodal_model='gmu'."
        )


def scaled_mse_loss(
    y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
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

    # Test MultimodalCrossAttentionDNN with current dataloader contract
    model4 = MultimodalCrossAttentionDNN(num_genes=250, dropout=0.4)
    y4 = model4(x3)
    assert (
        y4.shape == (2, 250)
    ), f"MultimodalCrossAttentionDNN output shape mismatch: {y4.shape}"
    print(f"✓ MultimodalCrossAttentionDNN output shape: {y4.shape}")

    # Test loss function
    y_hat = torch.randn(8, 250)
    y = torch.randn(8, 250)
    loss = scaled_mse_loss(y_hat, y)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ scaled_mse_loss: {loss.item():.4f}")
