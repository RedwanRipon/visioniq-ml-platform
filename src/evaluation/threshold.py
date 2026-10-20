import torch

from src.data.dataset import build_dataloaders
from src.evaluation.evaluate import predict
from src.models.resnet import build_model, get_device
from src.utils.config import PROJECT_ROOT, load_config

if __name__ == "__main__":
    config = load_config()
    device = get_device()
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "best_model.pth",
                                     map_location=device, weights_only=True))
    _, _, test_dl = build_dataloaders(config)
    y_true, y_pred, y_prob = predict(model, test_dl, device)
    confidence = y_prob.max(axis=1)
    for t in (0.0, 0.5, 0.7, 0.9, 0.95):
        answered = confidence >= t                       # images the model does NOT call "unknown"
        acc = (y_pred[answered] == y_true[answered]).mean()
        print(f"threshold {t:.2f}: answers {answered.mean():6.1%} of images, accuracy on those {acc:.2%}")