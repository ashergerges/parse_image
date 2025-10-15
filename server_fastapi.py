import io
import gc
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR ID Parser - Lightweight Arabic OCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def optimize_memory():
    """تحسين استخدام الذاكرة"""
    gc.collect()

def preprocess_image_lightweight(image):
    """معالجة خفيفة للصورة لتوفير الذاكرة"""
    
    # تحويل مباشر لتدرج رمادي
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image
    
    # تقليل حجم الصورة إذا كانت كبيرة
    max_size = 800
    if gray.size[0] > max_size:
        ratio = max_size / gray.size[0]
        new_size = (max_size, int(gray.size[1] * ratio))
        gray = gray.resize(new_size, Image.LANCZOS)
    
    # تحسين بسيط للتباين
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.5)
    
    return enhanced

@app.get("/")
async def home():
    return {"message": "🚀 Lightweight Arabic OCR API Running!"}

@app.post("/parse-image")
async def parse_image(file: UploadFile = File(...)):
    try:
        # تحرير الذاكرة قبل البدء
        optimize_memory()
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="الملف يجب أن يكون صورة")
        
        # قراءة الصورة
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        logger.info(f"📷 معالجة صورة: {image.size}")
        
        # معالجة خفيفة
        processed_image = preprocess_image_lightweight(image)
        
        # إعدادات Tesseract موفرة للذاكرة
        custom_config = '--oem 1 --psm 6 -c preserve_interword_spaces=1'
        
        # استخراج النص
        text = pytesseract.image_to_string(
            processed_image,
            lang='ara',
            config=custom_config
        )
        
        # تنظيف النص
        cleaned_text = " ".join(text.split())
        
        # تحرير الذاكرة بعد الانتهاء
        del image, processed_image, image_bytes
        optimize_memory()
        
        return {
            "success": True,
            "text": cleaned_text,
            "backend": "Tesseract-Arabic"
        }
        
    except Exception as e:
        logger.error(f"❌ خطأ: {e}")
        optimize_memory()
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الصورة: {str(e)}")
