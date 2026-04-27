import torch.nn as nn
from torchvision import models


def create_binary_emotion_model() -> nn.Module:
    """
    Build a transfer-learning model for binary emotion classification.
    """
    try:
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        backbone = models.resnet18(weights=None)

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )
    return backbone
