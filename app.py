from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import os

from predict import load_model, predict

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ load model ONCE
model = load_model()

@app.post("/predict")
async def run_inference(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

    with open(tmp.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(model, tmp.name)

    os.remove(tmp.name)

    return {"prediction": float(result)}