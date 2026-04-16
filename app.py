from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64

from predict import load_model, predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load once (already good)
model = load_model()

@app.post("/predict")
async def run_inference(file: UploadFile = File(...)):
    # 🚀 Read image directly (NO DISK I/O)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result, annotated, bin_detected = predict(model, img, return_image=True)

    # Encode output
    _, buffer = cv2.imencode('.jpg', annotated)
    img_str = base64.b64encode(buffer).decode()

    return {
        "prediction": float(result),
        "image": img_str,
        "bin_detected": bin_detected
    }