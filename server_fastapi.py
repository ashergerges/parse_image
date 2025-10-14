from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("⚠️ Environment variable HF_API_TOKEN is missing!")

MODEL_NAME = "microsoft/trocr-base-printed"
client = InferenceClient(model=MODEL_NAME, token=HF_API_TOKEN)

@app.post("/parse_image")
async def parse_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = client.image_to_text(image_bytes)
        return {"text": result.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)}")
