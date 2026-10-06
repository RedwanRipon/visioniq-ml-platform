import hashlib
import json
from collections import Counter

import numpy as np
from torchvision.datasets import CIFAR10

from src.data.preprocessing import CIFAR10_CLASSES
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def quality_report(images, labels, thresholds):
    """images: uint8 array (N, 32, 32, 3). Returns a dict with every check + PASS/WARNING/FAIL."""
    n = len(images)

    # 1. corrupted = not a 32x32 RGB uint8 image
    corrupted = sum(1 for img in images if img.shape != (32, 32, 3) or img.dtype != np.uint8)

    # 2. class distribution and imbalance (largest class / smallest class)
    counts = Counter(int(l) for l in labels)
    distribution = {CIFAR10_CLASSES[i]: counts.get(i, 0) for i in range(10)}
    imbalance = max(distribution.values()) / max(min(distribution.values()), 1)

    # 3. exact duplicates via SHA-256 of the raw pixel bytes
    hashes = [hashlib.sha256(img.tobytes()).hexdigest() for img in images]
    duplicates = n - len(set(hashes))

    # 4. channel statistics (pixel values scaled to 0-1)
    pixels = images.reshape(-1, 3) / 255.0
    channel_mean = pixels.mean(axis=0).round(4).tolist()
    channel_std = pixels.std(axis=0).round(4).tolist()

    # 5. outliers: brightness far from the average (z-score)
    brightness = images.reshape(n, -1).mean(axis=1)
    z = np.abs(brightness - brightness.mean()) / brightness.std()
    outliers = int((z > thresholds["outlier_z_threshold"]).sum())

    # 6. quality gate
    status = "PASS"
    if imbalance > thresholds["max_imbalance_ratio"]:
        status = "WARNING"
    if corrupted / n > thresholds["max_corruption_rate"] or duplicates / n > thresholds["max_duplicate_rate"]:
        status = "FAIL"

    return {"samples": n, "classes": len(distribution), "class_distribution": distribution,
            "imbalance_ratio": round(imbalance, 3), "corrupted": corrupted, "duplicates": duplicates,
            "channel_mean": channel_mean, "channel_std": channel_std, "outliers": outliers,
            "status": status}


def main():
    config = load_config()
    root = str(PROJECT_ROOT / config["data"]["data_dir"])
    reports = {}
    for split, is_train in (("train", True), ("test", False)):
        ds = CIFAR10(root, train=is_train, download=True)
        reports[split] = quality_report(ds.data, ds.targets, config["quality"])
        logger.info("%s: %s", split, {k: v for k, v in reports[split].items() if k != "class_distribution"})

    out = PROJECT_ROOT / "reports" / "data_quality_report.json"
    out.write_text(json.dumps(reports, indent=2))
    logger.info("saved %s", out)
    if any(r["status"] == "FAIL" for r in reports.values()):
        raise SystemExit("Data quality gate FAILED - stop the pipeline")


if __name__ == "__main__":
    main()