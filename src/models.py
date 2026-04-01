from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as torch_models


class ResNet(nn.Module):
    def __init__(self, input_channels: int, pretrained: bool, num_classes: int):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = torch_models.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = None

        self.base_model = torch_models.resnet18(weights=weights)
        self.base_model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class LeNet(nn.Module):
    def __init__(self, input_channels: int, pretrained: bool, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        feature_map_size = 5 if input_channels == 1 else 6
        self.fc1 = nn.Linear(16 * feature_map_size * feature_map_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sigmoid(self.conv1(x))
        x = self.pool(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model_class(name: str):
    name = name.lower()
    if name == "resnet":
        return ResNet
    if name == "lenet":
        return LeNet
    raise ValueError(f"Unsupported model name: {name}")


def initialize_model(model_class, input_channels: int, pretrained: bool, num_classes: int, seed: int, device: torch.device):
    cpu_state = torch.get_rng_state()
    gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model_class(input_channels, pretrained, num_classes).to(device)

    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and gpu_state is not None:
        torch.cuda.set_rng_state_all(gpu_state)

    return model
