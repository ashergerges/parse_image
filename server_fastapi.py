import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

app = FastAPI(title="OCR ID Parser - Tesseract Arabic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "🚀 Arabic OCR API Running Successfully!"}

@app.post("/parse-image")
async def parse_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # تحويل لتدرج رمادي
        gray = image.convert("L")

        # تحسين التباين والحدة
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        sharpened = enhanced.filter(ImageFilter.SHARPEN)

        # تحويل الصورة إلى أبيض وأسود (ثنائي)
        bw = sharpened.point(lambda x: 0 if x < 140 else 255, '1')

        # تشغيل OCR باستخدام Tesseract
        text = pytesseract.image_to_string(
            bw,
            lang="ara+eng",
            config="--psm 6"
        )

        cleaned = " ".join(text.split())
        return {"text": cleaned}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
