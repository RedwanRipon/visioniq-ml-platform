import torch
from src.utils.config import load_config
from src.models.resnet import build_model, get_device

device = get_device()
model = build_model(load_config()).to(device).eval()
x = torch.randn(4, 3, 32, 32, device=device)            # 4 fake images
with torch.no_grad():
    logits = model(x)
print("device:", device)
print("output shape:", tuple(logits.shape))             # expect (4, 10)
print("parameters:", sum(p.numel() for p in model.parameters()))
print("probabilities of image 0:", torch.softmax(logits[0], dim=0).round(decimals=3).tolist())