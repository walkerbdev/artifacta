"""ResNet model architecture."""

import torch.nn as nn
from torchvision import models


def create_resnet(num_classes=10, pretrained=True):
    """Create ResNet50 model for image classification."""
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
