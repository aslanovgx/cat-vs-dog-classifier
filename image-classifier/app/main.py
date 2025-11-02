# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image, ImageFile
import numpy as np
import io
import os

# ---- Böyük şəkillərdə parçalanmanı icazələndir ----
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- Yol konfiqurasiyası ----
APP_DIR = os.path.dirname(os.path.abspath(__file__))   # image-classifier/app
PROJECT_ROOT = os.path.dirname(APP_DIR)                # image-classifier/
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "cat_dog_model.h5")
INDEX_PATH = os.path.join(PROJECT_ROOT, "index.html")

IMG_SIZE = (150, 150)
THRESHOLD = 0.5
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# ---- FastAPI tətbiqi ----
app = FastAPI(title="Cat vs Dog Classifier")

# ---- Lokal test üçün CORS (istəyə görə silmək olar) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Modeli yüklə ----
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model load failed: {e}")

# ---- Kök route: index.html göstər və keşi söndür ----
@app.get("/", response_class=HTMLResponse)
def root():
    if not os.path.exists(INDEX_PATH):
        return HTMLResponse("<h1>index.html tapılmadı</h1>", status_code=404)

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # Safari və digər brauzerlər üçün keşləməni tam bağlayır
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

# ---- Şəkil preprocess funksiyası ----
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """bytes -> PIL RGB -> resize -> float32 [0,1] -> (1, H, W, 3)"""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

# ---- Proqnoz endpoint ----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    if not (file.content_type and file.content_type.startswith("image/")):
        raise HTTPException(status_code=415, detail="Only image files are accepted.")

    # Fayl ölçüsü limiti
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    x = preprocess_image(file_bytes)

    # Model proqnozu
    prob_dog = float(model.predict(x, verbose=0)[0][0])  # sigmoid: dog ehtimalı
    prob_cat = 1.0 - prob_dog
    label = "Dog" if prob_dog >= THRESHOLD else "Cat"

    return JSONResponse({
        "label": label,
        "probabilities": {"Cat": round(prob_cat, 4), "Dog": round(prob_dog, 4)},
        "threshold": THRESHOLD
    })

# ---- Health check ----
@app.get("/health")
def health():
    return {"ok": True}



# app/main.py faylının altına əlavə et:

import time
import os

@app.get("/__index_info")
def __index_info():
    exists = os.path.exists(INDEX_PATH)
    mtime = os.path.getmtime(INDEX_PATH) if exists else None
    head = ""
    if exists:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            head = f.read(200)
    return {
        "index_path": INDEX_PATH,
        "exists": exists,
        "mtime": mtime,
        "mtime_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)) if mtime else None,
        "head_preview": head
    }
