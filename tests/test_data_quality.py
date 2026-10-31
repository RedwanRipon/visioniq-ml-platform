import numpy as np

from src.data.quality import quality_report

THRESHOLDS = {"max_corruption_rate": 0.01, "max_duplicate_rate": 0.05,
              "max_imbalance_ratio": 1.5, "outlier_z_threshold": 4.0}


def fake_images(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)


def test_clean_balanced_data_passes():
    images = fake_images(20)
    labels = [i % 10 for i in range(20)]           # 2 images per class
    report = quality_report(images, labels, THRESHOLDS)
    assert report["samples"] == 20
    assert report["duplicates"] == 0
    assert report["status"] == "PASS"


def test_duplicates_are_detected():
    images = fake_images(20)
    images[5] = images[0]                          # make two exact copies
    images[9] = images[0]
    labels = [i % 10 for i in range(20)]
    report = quality_report(images, labels, THRESHOLDS)
    assert report["duplicates"] == 2
    assert report["status"] == "FAIL"              # 10% duplicates > 5% limit


def test_imbalance_gives_warning():
    images = fake_images(30)
    labels = [0] * 21 + list(range(1, 10))         # class 0 dominates
    report = quality_report(images, labels, THRESHOLDS)
    assert report["status"] == "WARNING"