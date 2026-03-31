import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class FashionEncoder(nn.Module):
    BACKBONE_OUT_DIM = 1280

    def __init__(self, embedding_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_OUT_DIM, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)
