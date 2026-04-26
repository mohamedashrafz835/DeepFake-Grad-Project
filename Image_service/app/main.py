from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch

from app.model import ModelLoader
from app.utils import preprocess

app = FastAPI(title="Forgery Detection API")

model = None

# 🔥 IMPORTANT: use threshold from training
THRESHOLD = 0.384


# =========================
# Load model at startup
# =========================
@app.on_event("startup")
def load_model():
    global model
    print("🚀 Loading model...")

    model = ModelLoader("/app/model.pth")
    model.model.eval()

    print("✅ Model loaded successfully")


# =========================
# Health check
# =========================
@app.get("/")
def health():
    return {
        "status": "running",
        "message": "Forgery Detection API is live"
    }


# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 🔥 safe image loading
        with Image.open(file.file) as img:
            img = img.convert("RGB")

            # preprocess (RGB + ELA)
            rgb, ela = preprocess(img)

            # inference
            output = model.predict(rgb, ela)

            # 🔥 convert logits → probabilities
            probs = torch.softmax(output, dim=1)

            prob_real = float(probs[0][0].item())
            prob_fake = float(probs[0][1].item())

            # 🔥 threshold decision (IMPORTANT FIX)
            pred = 1 if prob_fake > THRESHOLD else 0

        return {
            "prediction": int(pred),
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "threshold_used": THRESHOLD
        }

    except Exception as e:
        return {
            "error": str(e)
        }
