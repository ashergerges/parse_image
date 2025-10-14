from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import numpy as np
from PIL import Image
import io
import logging
import os

# تهيئة اللوجر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR API", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تهيئة EasyOCR مرة واحدة عند بدء التشغيل
reader = None

@app.on_event("startup")
async def startup_event():
    global reader
    try:
        logger.info("🚀 جاري تحميل نموذج EasyOCR...")
        # تحميل النموذج للعربية والإنجليزية
        reader = easyocr.Reader(['ar', 'en'], gpu=False)
        logger.info("✅ تم تحميل النموذج بنجاح!")
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل النموذج: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "OCR API is running!", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": reader is not None}

@app.post("/parse_image")
async def parse_image(file: UploadFile = File(...)):
    # التحقق من نوع الملف
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="الملف يجب أن يكون صورة")
    
    try:
        logger.info(f"📨 معالجة الصورة: {file.filename}")
        
        # قراءة بيانات الصورة
        image_data = await file.read()
        
        # تحويل إلى صورة PIL
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # تحويل إلى numpy array لـ EasyOCR
        image_np = np.array(image)
        
        # استخراج النص من الصورة
        results = reader.readtext(image_np)
        
        # تجميع النتائج
        if not results:
            return {"text": "", "confidence": 0, "message": "لم يتم العثور على نص"}
        
        # جمع كل النصوص المكتشفة
        all_text = " ".join([result[1] for result in results])
        
        # حساب متوسط الثقة
        confidences = [result[2] for result in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        logger.info(f"✅ تم استخراج النص بنجاح: {all_text[:50]}...")
        
        return {
            "text": all_text.strip(),
            "confidence": round(avg_confidence, 3),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"❌ خطأ في معالجة الصورة: {e}")
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الصورة: {str(e)}")

@app.post("/parse_image_simple")
async def parse_image_simple(file: UploadFile = File(...)):
    """نسخة مبسطة ترجع النص فقط"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
        
        results = reader.readtext(image_np)
        all_text = " ".join([result[1] for result in results])
        
        return {"text": all_text.strip()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ: {str(e)}")
