import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import re

app = FastAPI(title="OCR ID Parser - Tesseract Arabic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image):
    """معالجة مسبقة للصورة لتحسين دقة OCR للعربية"""
    
    # تحويل لتدرج رمادي
    gray = image.convert("L")
    
    # زيادة الدقة إذا كانت منخفضة
    if gray.size[0] < 500:
        new_size = (gray.size[0] * 2, gray.size[1] * 2)
        gray = gray.resize(new_size, Image.LANCZOS)
    
    # تحسين التباين
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.5)
    
    # تحسين الحدة
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    
    # إزالة الضوضاء
    denoised = sharpened.filter(ImageFilter.MedianFilter(size=3))
    
    # تحسين السطوع
    brightness_enhancer = ImageEnhance.Brightness(denoised)
    final_image = brightness_enhancer.enhance(1.2)
    
    return final_image

def postprocess_text(text):
    """تنظيف النص المستخرج"""
    # إزالة الأسطر الفارغة والمسافات الزائدة
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # تجميع النص مع الحفاظ على الترتيب
    cleaned_text = " ".join(lines)
    
    # إزالة الأحرف غير المرغوب فيها مع الحفاظ على العربية والأرقام
    cleaned_text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF0-9\s\-\.]', '', cleaned_text)
    
    # تقليل المسافات المتعددة
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

@app.get("/")
async def home():
    return {"message": "🚀 Arabic OCR API Running Successfully!"}

@app.post("/parse-image")
async def parse_image(file: UploadFile = File(...)):
    try:
        # التحقق من نوع الملف
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="الملف يجب أن يكون صورة")
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # معالجة مسبقة للصورة
        processed_image = preprocess_image(image)
        
        # إعدادات Tesseract المخصصة للعربية
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=٠١٢٣٤٥٦٧٨٩0123456789ابجدهوزحطيكلمنسعفصقرشتثخذضظغءآأؤإئةىﻻﻷﻹﻵ\u0600-\u06FF\s\-'
        
        # تشغيل OCR مع إعدادات مخصصة
        text = pytesseract.image_to_string(
            processed_image,
            lang='ara+eng',
            config=custom_config
        )
        
        # تنظيف النص
        cleaned_text = postprocess_text(text)
        
        return {
            "success": True,
            "text": cleaned_text,
            "original_length": len(text),
            "cleaned_length": len(cleaned_text)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الصورة: {str(e)}")

# نقطة نهاية جديدة لاختبار Tesseract
@app.get("/test-tesseract")
async def test_tesseract():
    """اختبار تثبيت Tesseract واللغات المتاحة"""
    try:
        # الحصول على اللغات المتاحة
        languages = pytesseract.get_languages()
        
        # اختبار بسيط
        test_image = Image.new('RGB', (100, 100), color='white')
        test_result = pytesseract.image_to_string(test_image, lang='ara')
        
        return {
            "available_languages": languages,
            "tesseract_version": pytesseract.get_tesseract_version(),
            "test_result": "Tesseract يعمل بشكل صحيح" if test_result is not None else "مشكلة في Tesseract"
        }
    except Exception as e:
        return {"error": str(e)}
