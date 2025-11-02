# ---- Python bazası ----
FROM python:3.11-slim

# ---- Sistem asılılıqları (Pillow, numpy üçün) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo zlib1g libpng16-16 libopenblas0 \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# ---- İş qovluğu ----
WORKDIR /app

# ---- Asılılıqları quraşdır ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Layihə fayllarını köçür ----
COPY . .

# ---- Default port ----
# EXPOSE 8000

EXPOSE 7860
ENV PORT=7860

# ---- Başlama əmri ----
# CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]