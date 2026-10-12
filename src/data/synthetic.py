import torch
from torchvision import transforms

from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD


class AddGaussianNoise:
    """Adds random noise to a tensor image half of the time."""
    def __init__(self, std=0.03):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std if torch.rand(1).item() < 0.5 else x


def build_train_transform(policy="standard"):
    if policy == "none":
        ops, tensor_ops = [], []
    elif policy == "standard":
        ops = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
               transforms.RandomHorizontalFlip(),
               transforms.ColorJitter(0.2, 0.2, 0.2)]
        tensor_ops = [transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))]
    elif policy == "strong":
        ops = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
               transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(15),
               transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
               transforms.ColorJitter(0.4, 0.4, 0.4, 0.05)]
        tensor_ops = [AddGaussianNoise(0.03), transforms.RandomErasing(p=0.5, scale=(0.02, 0.25))]
    else:
        raise ValueError(f"unknown policy {policy}")
    return transforms.Compose([*ops, transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), *tensor_ops])