# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_400_BAD_REQUEST
from tensorflow.keras.models import load_model
from PIL import Image, ImageFile
import numpy as np
import io
import os
import time

# ---- Model və sxemlər ----
from app.models import PredictionOut, ErrorOut

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

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# ---- FastAPI tətbiqi ----
app = FastAPI(
    title="Cat vs Dog Classifier",
    description="FastAPI + TensorFlow ilə sadə pişik/it təsnifatı. Şəkil yüklə, etiket və ehtimalları al.",
    version="1.0.0",
    contact={"name": "Mustafa Aslanov", "email": "mustafa.aslanovv@gmail.com"},
)

# ---- Lokal test üçün CORS (istəyə görə silmək olar) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔽 CORS-DAN SONRA BUNU ƏLAVƏ ET
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.max_bytes:
            return JSONResponse(
                status_code=HTTP_400_BAD_REQUEST,
                content={"detail": f"File too large. Max {self.max_bytes // (1024*1024)}MB"},
            )
        return await call_next(request)

# Middleware-i aktivləşdir
app.add_middleware(LimitUploadSizeMiddleware, max_bytes=MAX_UPLOAD_BYTES)


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
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data.")

# ---- Exception handler-lər ----
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": exc.errors()},
    )

# ---- Proqnoz endpoint ----
@app.post(
    "/predict",
    response_model=PredictionOut,
    responses={
        400: {"model": ErrorOut},
        413: {"model": ErrorOut},
        415: {"model": ErrorOut},
    },
)
async def predict(file: UploadFile = File(...)):
    # Fayl adı və tipi yoxlanır
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG/PNG/WebP images are accepted.")


    # Fayl ölçüsü limiti
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    # Şəkli preprocess et
    x = preprocess_image(file_bytes)

    # Model proqnozu
    prob_dog = float(model.predict(x, verbose=0)[0][0])  # sigmoid: dog ehtimalı
    prob_cat = 1.0 - prob_dog
    label = "Dog" if prob_dog >= THRESHOLD else "Cat"

    # JSON cavabı
    return {
        "label": label,
        "probabilities": {"Cat": round(prob_cat, 4), "Dog": round(prob_dog, 4)},
        "threshold": THRESHOLD,
    }

# ---- Health check ----
@app.get("/health")
def health():
    return {"ok": True}

# ---- Faylın vəziyyəti: index.html haqqında info ----
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
