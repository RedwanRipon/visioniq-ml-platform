import argparse
import time
import mlflow
import json


import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from datetime import datetime, timezone

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
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run-name", default="resnet18")
    parser.add_argument("--policy", choices=["none", "standard", "strong"], default="standard")
    parser.add_argument("--params", default=None, help="JSON file with tuned hyperparameters")
    args = parser.parse_args()

    config = load_config()
    if args.params:
        best = json.loads((PROJECT_ROOT / args.params).read_text())
        config["training"]["learning_rate"] = best["learning_rate"]
        config["training"]["weight_decay"] = best["weight_decay"]
        config["training"]["dropout"] = best["dropout"]
        config["data"]["batch_size"] = best["batch_size"]

    set_seed(config["project"]["seed"])
    device = get_device()
    epochs = args.epochs or config["training"]["epochs"]

    train_dl, val_dl, _ = build_dataloaders(config, args.policy)
    if args.subset:                                 # smoke test: tiny slice of the data
        train_dl = DataLoader(Subset(train_dl.dataset, range(args.subset)),
                              batch_size=config["data"]["batch_size"], shuffle=True)

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = args.lr or config["training"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 

    best_acc = 0.0
    ckpt_path = PROJECT_ROOT / "models" / f"{args.run_name}.pth"
    (PROJECT_ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({"model": config["training"]["model"], "learning_rate": lr,
                           "weight_decay": config["training"]["weight_decay"], "epochs": epochs,
                           "batch_size": config["data"]["batch_size"], "optimizer": "AdamW",
                           "dropout": config["training"].get("dropout", 0.2),
                           "seed": config["project"]["seed"], "train_images": len(train_dl.dataset),"augmentation": args.policy})
        mlflow.log_artifact(str(PROJECT_ROOT / "configs" / "config.yaml"))

        logger.info("device=%s epochs=%d lr=%g", device, epochs, lr)
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_dl, criterion, device)
            scheduler.step()
            logger.info("epoch %d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f | %.0fs",
                        epoch, train_loss, train_acc, val_loss, val_acc, time.time() - start)
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc,
                                "validation_loss": val_loss, "validation_accuracy": val_acc}, step=epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                logger.info("saved best model (val_acc=%.4f)", val_acc)

        mlflow.log_metric("best_validation_accuracy", best_acc)
        mlflow.log_artifact(str(ckpt_path))
        metadata = {
            "model_name": "resnet18", "model_version": "1.0.0",
            "dataset": "cifar10", "classes": 10, "augmentation": args.policy,
            "training_config": {"learning_rate": lr, "epochs": epochs,
                                "batch_size": config["data"]["batch_size"],
                                "weight_decay": config["training"]["weight_decay"],
                                "dropout": config["training"].get("dropout", 0.2),
                                "seed": config["project"]["seed"]},
            "metrics": {"best_validation_accuracy": best_acc},
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (PROJECT_ROOT / "models" / f"{args.run_name}_metadata.json").write_text(json.dumps(metadata, indent=2))          # the model file is stored with the run


if __name__ == "__main__":
    main()