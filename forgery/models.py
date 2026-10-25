import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        side = (size - 8) // 2
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32 * side * side, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name, size=128, pretrained=True):
    weights = "DEFAULT" if pretrained else None
    if name == "small_cnn":
        return SmallCNN(size)
    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    else:
        raise ValueError(f"unknown model: {name}")
    return model
