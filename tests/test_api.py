import io

from fastapi.testclient import TestClient
from PIL import Image

from api.app import app


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    with TestClient(app) as client:                # also runs lifespan, i.e. loads the model
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "healthy"}


def test_valid_image_returns_prediction():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert len(body["top_k"]) == 3
        assert 0.0 <= body["confidence"] <= 1.0


def test_text_file_is_rejected():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
        assert r.status_code == 415


def test_corrupted_image_is_rejected():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.png", b"not really a png", "image/png")})
        assert r.status_code == 400