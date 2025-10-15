from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import numpy as np
from PIL import Image
import io
import logging

# إعداد اللوج
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_fastapi")

app = FastAPI(title="OCR API (Optimized)", version="1.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ تأجيل تحميل النموذج حتى أول طلب
reader = None

def get_reader():
    """تحميل النموذج عند أول استخدام فقط لتوفير الذاكرة."""
    global reader
    if reader is None:
        logger.info("🚀 تحميل EasyOCR لأول مرة (قد يستغرق دقيقة)...")
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            logger.info("✅ تم تحميل نموذج EasyOCR بنجاح!")
        except Exception as e:
            logger.error(f"❌ فشل تحميل النموذج: {e}")
            raise HTTPException(status_code=500, detail="فشل تحميل نموذج OCR")
    return reader


@app.get("/")
async def root():
    return {"message": "OCR API is running!", "status": "active"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": reader is not None}


@app.post("/parse_image")
async def parse_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="الملف يجب أن يكون صورة")

    try:
        logger.info(f"📨 استقبال صورة: {file.filename}")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        # تحميل القارئ عند الطلب فقط
        ocr = get_reader()

        # تشغيل OCR
        results = ocr.readtext(image_np)

        if not results:
            return {"text": "", "confidence": 0, "message": "لم يتم العثور على نص"}

        all_text = " ".join([r[1] for r in results])
        confidences = [r[2] for r in results]
        avg_conf = sum(confidences) / len(confidences)

        logger.info(f"✅ OCR Done: {all_text[:60]}...")
        return {
            "text": all_text.strip(),
            "confidence": round(avg_conf, 3),
            "filename": file.filename,
        }

    except MemoryError:
        logger.error("❌ نفاد الذاكرة أثناء معالجة الصورة!")
        raise HTTPException(status_code=500, detail="نفاد الذاكرة أثناء المعالجة")
    except Exception as e:
        logger.error(f"❌ خطأ أثناء معالجة الصورة: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse_image_simple")
async def parse_image_simple(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        ocr = get_reader()
        results = ocr.readtext(np.array(image))
        all_text = " ".join([r[1] for r in results])
        return {"text": all_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
