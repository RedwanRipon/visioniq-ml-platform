from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../visioniq-ml-platform


def load_config(path=None):
    """Read configs/config.yaml and return it as a Python dict."""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)