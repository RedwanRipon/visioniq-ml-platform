import json

import matplotlib
matplotlib.use("Agg")                               # save plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.data.dataset import build_dataloaders
from src.data.preprocessing import CIFAR10_CLASSES
from src.models.resnet import build_model, get_device
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)
REPORTS = PROJECT_ROOT / "reports"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        y_prob.append(probs.cpu().numpy())
        y_true.append(labels.numpy())
    y_prob = np.concatenate(y_prob)
    return np.concatenate(y_true), y_prob.argmax(1), y_prob


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc_ovr": roc_auc_score(y_true, y_prob, multi_class="ovr"),
    }


def sanity_checks(model, device):
    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    probs = torch.softmax(out, dim=1)
    return {
        "output_shape_is_4x10": tuple(out.shape) == (4, 10),
        "outputs_are_finite": bool(torch.isfinite(out).all()),
        "probabilities_sum_to_1": bool(torch.allclose(probs.sum(1), torch.ones(4, device=device))),
        "ten_class_names": len(CIFAR10_CLASSES) == 10,
    }


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CIFAR10_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    config = load_config()
    device = get_device()
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "best_model.pth",
                                     map_location=device, weights_only=True))

    checks = sanity_checks(model, device)
    logger.info("sanity checks: %s", checks)
    if not all(checks.values()):
        raise SystemExit("Sanity checks failed - do not use this model")

    _, _, test_dl = build_dataloaders(config)
    y_true, y_pred, y_prob = predict(model, test_dl, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, digits=4))
    save_confusion_matrix(confusion_matrix(y_true, y_pred), REPORTS / "confusion_matrix.png")
    logger.info("saved reports to %s", REPORTS)


if __name__ == "__main__":
    main()