import io
import gc
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR ID Parser - Enhanced Arabic OCR")

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

def enhance_image_for_arabic_text(image):
    """تحسين متقدم للصورة مخصص للنصوص العربية في البطاقات"""
    
    # تحويل PIL إلى OpenCV
    img_array = np.array(image)
    
    # إذا كانت الصورة ملونة، تحويلها إلى تدرج رمادي
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # زيادة الدقة لتحسين دقة OCR
    scale_factor = 2
    new_width = gray.shape[1] * scale_factor
    new_height = gray.shape[0] * scale_factor
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # تطبيق تصفية لتحسين الحدة
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    # تحسين التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(sharpened)
    
    # تحويل الظلال إلى أبيض نقي والخلفية إلى سوداء
    _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # تحويل kembali ke PIL Image
    result_image = Image.fromarray(binary)
    
    return result_image

def preprocess_multiple_techniques(image):
    """تجربة تقنيات متعددة للمعالجة المسبقة"""
    
    techniques = []
    
    # التقنية 1: تحسين أساسي
    gray = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    basic_enhanced = enhancer.enhance(2.0)
    techniques.append(basic_enhanced)
    
    # التقنية 2: تحسين متقدم
    advanced_enhanced = enhance_image_for_arabic_text(image)
    techniques.append(advanced_enhanced)
    
    # التقنية 3: تقليل الضوضاء
    denoised = gray.filter(ImageFilter.MedianFilter(size=3))
    enhancer_denoise = ImageEnhance.Contrast(denoised)
    denoise_enhanced = enhancer_denoise.enhance(2.5)
    techniques.append(denoise_enhanced)
    
    return techniques

def clean_arabic_text(text):
    """تنظيف متقدم للنص العربي"""
    if not text:
        return ""
    
    # إزالة الرموز غير المرغوب فيها مع الحفاظ على العربية والأرقام
    arabic_pattern = r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF0-9\s\-\.\/]'
    cleaned = re.sub(arabic_pattern, '', text)
    
    # تصحيح المسافات
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # تصحيح الأرقام العربية
    number_corrections = {
        '٠': '٠', '۰': '٠',  # صفر عربي
        '١': '١', '۱': '١',  # واحد
        '٢': '٢', '۲': '٢',  # اثنان
        '٣': '٣', '۳': '٣',  # ثلاثة
        '٤': '٤', '۴': '٤',  # أربعة
        '٥': '٥', '۵': '٥',  # خمسة
        '٦': '٦', '۶': '٦',  # ستة
        '٧': '٧', '۷': '٧',  # سبعة
        '٨': '٨', '۸': '٨',  # ثمانية
        '٩': '٩', '۹': '٩',  # تسعة
    }
    
    for wrong, correct in number_corrections.items():
        cleaned = cleaned.replace(wrong, correct)
    
    # فصل الأسطر بشكل صحيح
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    
    # تجميع النص مع الحفاظ على الهيكل
    final_text = "\n".join(lines)
    
    return final_text.strip()

@app.get("/")
async def home():
    return {"message": "🚀 Enhanced Arabic OCR API Running!"}

@app.post("/parse-image")
async def parse_image(file: UploadFile = File(...)):
    try:
        optimize_memory()
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="الملف يجب أن يكون صورة")
        
        # قراءة الصورة
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        logger.info(f"📷 معالجة صورة: {image.size} - {image.mode}")
        
        # تجربة تقنيات متعددة للمعالجة المسبقة
        processed_images = preprocess_multiple_techniques(image)
        
        all_results = []
        
        for i, processed_img in enumerate(processed_images):
            logger.info(f"🔍 تجربة التقنية {i+1}")
            
            # إعدادات Tesseract مختلفة لكل تقنية
            configs = [
                '--oem 3 --psm 6 -c tessedit_char_whitelist=٠١٢٣٤٥٦٧٨٩0123456789ابجدهوزحطيكلمنسعفصقرشتثخذضظغءآأؤإئةى',
                '--oem 3 --psm 4',  # PSM 4 للكتلة الواحدة من النص
                '--oem 3 --psm 11',  # PSM 11 للنص الخالص
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(
                        processed_img,
                        lang='ara+eng',
                        config=config
                    )
                    
                    cleaned_text = clean_arabic_text(text)
                    if cleaned_text and len(cleaned_text) > 10:  # تجاهل النتائج القصيرة
                        all_results.append({
                            'technique': f'tech_{i+1}_config_{configs.index(config)+1}',
                            'text': cleaned_text,
                            'length': len(cleaned_text)
                        })
                        
                except Exception as e:
                    logger.warning(f"⚠️ فشل في التقنية {i+1}: {e}")
                    continue
        
        # اختيار أفضل نتيجة (الأطول عادةً)
        if all_results:
            best_result = max(all_results, key=lambda x: x['length'])
            final_text = best_result['text']
            logger.info(f"✅ أفضل نتيجة: {best_result['technique']} - طول: {best_result['length']}")
        else:
            final_text = ""
            logger.warning("⚠️ لم يتم العثور على نص")
        
        # تحرير الذاكرة
        del image, processed_images, image_bytes
        optimize_memory()
        
        return {
            "success": True,
            "text": final_text,
            "backend": "Tesseract-Arabic-Enhanced",
            "techniques_tried": len(processed_images),
            "results_found": len(all_results)
        }
        
    except Exception as e:
        logger.error(f"❌ خطأ: {e}")
        optimize_memory()
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الصورة: {str(e)}")

@app.post("/parse-image-detailed")
async def parse_image_detailed(file: UploadFile = File(...)):
    """إرجاع نتائج مفصلة من جميع التقنيات"""
    try:
        optimize_memory()
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="الملف يجب أن يكون صورة")
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        processed_images = preprocess_multiple_techniques(image)
        
        detailed_results = []
        
        for i, processed_img in enumerate(processed_images):
            configs = [
                ('psm6', '--oem 3 --psm 6'),
                ('psm4', '--oem 3 --psm 4'),
                ('psm11', '--oem 3 --psm 11'),
            ]
            
            for config_name, config in configs:
                try:
                    text = pytesseract.image_to_string(
                        processed_img,
                        lang='ara+eng',
                        config=config
                    )
                    
                    cleaned_text = clean_arabic_text(text)
                    
                    if cleaned_text and len(cleaned_text) > 5:
                        detailed_results.append({
                            'technique': f'tech_{i+1}',
                            'config': config_name,
                            'text': cleaned_text,
                            'length': len(cleaned_text)
                        })
                        
                except Exception as e:
                    continue
        
        # تحرير الذاكرة
        del image, processed_images, image_bytes
        optimize_memory()
        
        return {
            "success": True,
            "all_results": detailed_results,
            "best_result": max(detailed_results, key=lambda x: x['length']) if detailed_results else None
        }
        
    except Exception as e:
        optimize_memory()
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الصورة: {str(e)}")
