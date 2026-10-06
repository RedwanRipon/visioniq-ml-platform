import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from src.data.preprocessing import eval_transform, train_transform
from src.utils.config import PROJECT_ROOT 
from  src.utils.logging import get_logger

logger = get_logger(__name__)


def build_dataloaders(config):
    root = str(PROJECT_ROOT / config["data"]["data_dir"])
    seed = config["project"]["seed"]
    bs = config["data"]["batch_size"]
    nw = config["data"]["num_workers"]

    # Same 50k images loaded twice: once with training transforms, once without.
    train_full = CIFAR10(root, train=True, download=True, transform=train_transform())
    val_full = CIFAR10(root, train=True, download=True, transform=eval_transform())
    test_set = CIFAR10(root, train=False, download=True, transform=eval_transform())

    # Seeded shuffle of the indices 0..49999, then cut 10% off for validation.
    n_val = int(len(train_full) * config["data"]["validation_fraction"])
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_set = Subset(train_full, perm[n_val:])
    val_set = Subset(val_full, perm[:n_val])
    logger.info("train=%d val=%d test=%d", len(train_set), len(val_set), len(test_set))

    return (
        DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw),
        DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=nw),
    )