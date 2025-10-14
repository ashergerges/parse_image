from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import os

app = FastAPI()

# ✅ CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد نطاقك فقط لاحقًا
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Hugging Face Token (from Render environment variable)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("⚠️ Environment variable HF_API_TOKEN is missing!")

MODEL_NAME = "microsoft/trocr-base-printed"  # OCR Model
client = InferenceClient(model=MODEL_NAME, token=HF_API_TOKEN)

@app.post("/parse_image")
async def parse_image(file: UploadFile = File(...)):
    try:
        # 1️⃣ اقرأ الصورة كـ bytes
        image_bytes = await file.read()

        # 2️⃣ أرسلها للموديل
        result = client.image_to_text(data=image_bytes)

        # 3️⃣ رجع النص
        return {"text": result.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")
