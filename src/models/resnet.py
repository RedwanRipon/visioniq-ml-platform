import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class VisionIQResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, dropout=0.2, input_size=128):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)                    # ImageNet knowledge
        in_features = self.backbone.fc.in_features                   # 512
        self.backbone.fc = nn.Sequential(                            # new head: 512 -> 10
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.input_size = input_size

    def forward(self, x):
        # x: (batch, 3, 32, 32) -> upsample -> (batch, 3, 128, 128) -> backbone -> (batch, 10)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.backbone(x)


def build_model(config):
    return VisionIQResNet(input_size=config["data"]["image_size"])


def get_device():
    """Use the GPU if there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")