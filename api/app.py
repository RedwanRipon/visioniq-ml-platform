import io
import json
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from api.schemas import Prediction, TopK

from api.schemas import Prediction
from src.data.preprocessing import CIFAR10_CLASSES, eval_transform
from src.models.resnet import VisionIQResNet
from src.utils.config import PROJECT_ROOT

MAX_BYTES = 5 * 1024 * 1024                                   # 5 MB upload limit
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
state = {}


@asynccontextmanager
async def lifespan(app):
    # runs ONCE when the server starts
    meta = json.loads((PROJECT_ROOT / "models" / "best_model_metadata.json").read_text())
    model = VisionIQResNet(pretrained=False, dropout=meta["training_config"]["dropout"])
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "best_model.pth",
                                     map_location="cpu", weights_only=True))
    state["model"] = model.eval()
    state["meta"] = meta
    yield


app = FastAPI(title="VisionIQ API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    meta = state["meta"]
    return {"model_name": meta["model_name"], "model_version": meta["model_version"],
            "dataset": meta["dataset"], "classes": CIFAR10_CLASSES}


@app.post("/predict", response_model=Prediction, response_model_by_alias=True)
async def predict(file: UploadFile = File(...)):
    # --- 1. validate the upload ---
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported file type {file.content_type}")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File larger than 5 MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()                                          # forces a full decode
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "File is not a valid image")

    # --- 2. preprocess exactly like validation, then 3. predict ---
    start = time.perf_counter()
    x = eval_transform()(image).unsqueeze(0)                  # (1, 3, 32, 32)
    with torch.no_grad():
        probs = torch.softmax(state["model"](x), dim=1)[0]
    top_p, top_i = probs.topk(3)
    elapsed = (time.perf_counter() - start) * 1000

    return Prediction(
        predicted_class=CIFAR10_CLASSES[top_i[0]],
        confidence=round(top_p[0].item(), 4),
        top_k=[TopK(class_name=CIFAR10_CLASSES[i], probability=round(p.item(), 4))
               for p, i in zip(top_p, top_i)],
        inference_time_ms=round(elapsed, 2),
        model_version=state["meta"]["model_version"],
    )