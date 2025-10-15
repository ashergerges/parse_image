# استخدم نسخة Python الرسمية
FROM python:3.11-slim

# تثبيت Tesseract + تحميل نموذج اللغة العربية الكامل (tessdata_best)
RUN apt-get update && \
    apt-get install -y tesseract-ocr wget && \
    mkdir -p /usr/share/tesseract-ocr/4.00/tessdata && \
    wget -O /usr/share/tesseract-ocr/4.00/tessdata/ara.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/ara.traineddata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# تحديد مجلد العمل
WORKDIR /app

# نسخ ملفات المشروع
COPY . /app

# تثبيت باقات Python
RUN pip install --no-cache-dir -r requirements.txt

# فتح المنفذ
EXPOSE 8000

# تشغيل التطبيق
CMD ["uvicorn", "server_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
