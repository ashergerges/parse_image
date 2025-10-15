import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import pytesseract
import logging

app = FastAPI(title="OCR Arabic ID Reader - Tesseract")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_fastapi")

@app.get("/")
async def root():
    return {"message": "🚀 Arabic OCR API Running!"}

@app.post("/parse_image")
async def parse_image(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 Received image: {file.filename}")
        image_bytes = await file.read()

        # فتح الصورة
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # تحويل لتدرج رمادي + زيادة الوضوح
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)

        # تحسين القراءة عبر threshold بسيط
        gray_np = np.array(gray)
        gray_np = np.where(gray_np > 160, 255, 0).astype(np.uint8)
        gray = Image.fromarray(gray_np)

        # قراءة النص بالعربية فقط
        text = pytesseract.image_to_string(gray, lang="ara")

        text_clean = text.strip()
        logger.info(f"✅ OCR Done. Extracted text length: {len(text_clean)}")

        return {"text": text_clean}

    except Exception as e:
        logger.exception("❌ Error during OCR:")
        raise HTTPException(status_code=500, detail=str(e))
