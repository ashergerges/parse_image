from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import easyocr
from PIL import Image
import io
import numpy as np
import logging

# ==========================
# إعداد السيرفر
# ==========================
app = FastAPI(title="OCR ID Parser - EasyOCR Arabic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_fastapi")

# تهيئة EasyOCR مرة واحدة عند بدء التشغيل
reader = None

@app.on_event("startup")
async def startup_event():
    global reader
    try:
        logger.info("🚀 جاري تحميل نموذج EasyOCR...")
        reader = easyocr.Reader(['ar', 'en'], gpu=False)
        logger.info("✅ تم تحميل النموذج بنجاح!")
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل النموذج: {e}")
        raise e

# ==========================
# نقطة الفحص
# ==========================
@app.get("/")
async def home():
    return {"message": "🚀 OCR API Running with EasyOCR!"}

# ==========================
# رفع الصورة وتحليلها
# ==========================
@app.post("/parse-image")
async def parse_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="الملف يجب أن يكون صورة")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # تحويل الصورة لـ numpy array
        image_np = np.array(image)

        # استخراج النص
        results = reader.readtext(image_np, paragraph=True)

        if not results:
            return {"text": ""}

        # دمج النصوص المكتشفة
        all_text = " ".join([res[1] for res in results])
        return {"text": all_text.strip()}

    except Exception as e:
        logger.exception("❌ Error during OCR:")
        raise HTTPException(status_code=500, detail=str(e))
