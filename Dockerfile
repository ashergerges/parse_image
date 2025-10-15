FROM python:3.11-slim

# تثبيت Tesseract فقط مع اللغة العربية (بدون كل الحزم)
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-ara && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "server_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
