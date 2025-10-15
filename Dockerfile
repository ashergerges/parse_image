FROM python:3.11-slim

# تثبيت Tesseract مع حزم اللغة العربية الكاملة
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libtesseract-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# تحميل بيانات اللغة العربية عالية الجودة
RUN wget -O /usr/share/tesseract-ocr/4.00/tessdata/ara.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/ara.traineddata

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "server_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
