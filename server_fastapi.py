import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import numpy as np
import logging

# ==========================
# إعداد السيرفر
# ==========================
app = FastAPI(title="OCR ID Parser - Tesseract")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # أو ضع نطاقك فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_fastapi")

# ==========================
# نقطة الفحص
# ==========================
@app.get("/")
async def home():
    return {"message": "🚀 OCR API Running with Tesseract!"}

# ==========================
# نقطة رفع الصورة وتحليلها
# ==========================
@app.post("/parse-image")
async def parse_image(file: UploadFile = File(...)):
    try:
        logger.info("📥 Received image: %s", file.filename)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # تحويل الصورة لـ numpy وتحسينها قبل OCR
        img = np.array(image.convert("L"))  # تدرج رمادي
        text = pytesseract.image_to_string(img, lang="ara+eng")

        logger.info("✅ OCR Done. Extracted text length: %d", len(text))
        return {"text": text.strip()}

    except Exception as e:
        logger.exception("❌ Error during OCR:")
        raise HTTPException(status_code=500, detail=str(e))
