from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import os
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

model = load_model()

@app.post("/predict")
async def run_inference(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

    with open(tmp.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result, annotated, bin_detected = predict(model, tmp.name, return_image=True)

    _, buffer = cv2.imencode('.jpg', annotated)
    img_str = base64.b64encode(buffer).decode()

    os.remove(tmp.name)

    return {
        "prediction": float(result),
        "image": img_str,
        "bin_detected": bool(bin_detected)
    }