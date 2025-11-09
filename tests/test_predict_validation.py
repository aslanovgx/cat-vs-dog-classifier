import io
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app.main as app_module  # app, MAX_UPLOAD_BYTES, model buradadır

client = TestClient(app_module.app)

def test_predict_get_is_not_allowed():
    r = client.get("/predict")
    assert r.status_code == 405  # Method Not Allowed

def test_predict_rejects_non_image():
    fake = io.BytesIO(b"not-an-image")
    files = {"file": ("test.txt", fake, "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 415  # Only JPEG/PNG/WebP images are accepted.

def test_predict_accepts_image_and_returns_schema(monkeypatch):
    # 1) Kiçik 150x150 PNG şəkli yaradıb yaddaşa yaz
    img = Image.new("RGB", (150, 150), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 2) Modeli yüngül "fake" predict ilə əvəz edirik ki, test sürətli olsun
    class FakeModel:
        def predict(self, x, verbose=0):
            # 0.9 → Dog label
            return np.array([[0.9]], dtype="float32")

    monkeypatch.setattr(app_module, "model", FakeModel())

    files = {"file": ("dummy.png", buf, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200

    data = r.json()
    assert "label" in data and data["label"] == "Dog"
    assert "probabilities" in data
    assert "Cat" in data["probabilities"] and "Dog" in data["probabilities"]
    assert "threshold" in data
    # ehtimallar 0..1 aralığında
    assert 0.0 <= data["probabilities"]["Dog"] <= 1.0
    assert 0.0 <= data["probabilities"]["Cat"] <= 1.0

def test_predict_too_large_file():
    # MAX_UPLOAD_BYTES + 1 ölçüdə saxta fayl (başlıqda Content-Length görsənə bilər)
    big = io.BytesIO(b"\x00" * (app_module.MAX_UPLOAD_BYTES + 1))
    files = {"file": ("big.png", big, "image/png")}
    r = client.post("/predict", files=files)
    # middleware 400 qaytara bilər, endpoint 413 — hər ikisi qəbul ediləndir
    assert r.status_code in (400, 413)
