import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import build_dataloaders
from src.models.resnet import build_model, get_device
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                                   # dropout ON
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                       # forget last batch's gradients
        outputs = model(images)                     # 1. forward
        loss = criterion(outputs, labels)           # 2. how wrong?
        loss.backward()                             # 3. gradients
        optimizer.step()                            # 4. update weights
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()                                    # no gradients needed
def evaluate(model, loader, criterion, device):
    model.eval()                                    # dropout OFF
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None, help="use only N training images (quick test)")
    args = parser.parse_args()

    config = load_config()
    set_seed(config["project"]["seed"])
    device = get_device()
    epochs = args.epochs or config["training"]["epochs"]

    train_dl, val_dl, _ = build_dataloaders(config)
    if args.subset:                                 # smoke test: tiny slice of the data
        train_dl = DataLoader(Subset(train_dl.dataset, range(args.subset)),
                              batch_size=config["data"]["batch_size"], shuffle=True)

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    ckpt_path = PROJECT_ROOT / "models" / "best_model.pth"
    logger.info("device=%s epochs=%d train_images=%d", device, epochs, len(train_dl.dataset))
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step()
        logger.info("epoch %d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f | %.0fs",
                    epoch, train_loss, train_acc, val_loss, val_acc, time.time() - start)
        if val_acc > best_acc:                      # checkpoint only when it improves
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            logger.info("saved best model (val_acc=%.4f) -> %s", val_acc, ckpt_path)


if __name__ == "__main__":
    main()