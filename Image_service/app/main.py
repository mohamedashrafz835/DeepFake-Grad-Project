import os
import boto3
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import numpy as np
import cv2
import base64

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from app.model import ModelLoader
from app.utils import preprocess

app = FastAPI(title="Forgery Detection API")

model = None
THRESHOLD = 0.384
LOCAL_MODEL_PATH = "/app/model.pth"


# =========================
# S3 helper
# =========================
def _parse_s3_uri(s3_uri: str):
    """Parse 's3://bucket/key/path' → (bucket, key)."""
    s3_uri = s3_uri.replace("s3://", "", 1)
    bucket, _, key = s3_uri.partition("/")
    return bucket, key


def _download_from_s3(s3_path: str, local_path: str) -> None:
    print(f"⬇️  Downloading model from {s3_path} …")
    bucket, key = _parse_s3_uri(s3_path)
    boto3.client("s3").download_file(bucket, key, local_path)
    print(f"✅ Model downloaded to {local_path}")


# =========================
# Wrapper for Grad-CAM
# =========================
class WrapperModel(torch.nn.Module):
    def __init__(self, model, ela):
        super().__init__()
        self.model = model
        self.ela = ela

    def forward(self, x):
        return self.model(x, self.ela)


# =========================
# Load model once at startup
# =========================
@app.on_event("startup")
def load_model():
    global model
    s3_path = os.environ.get("S3_MODEL_PATH", "").strip()

    if s3_path:
        # Production: pull weights from S3 (path provided by Lambda → CI/CD)
        _download_from_s3(s3_path, LOCAL_MODEL_PATH)
    elif os.path.exists(LOCAL_MODEL_PATH):
        # Local dev: use pre-existing weights
        print(f"ℹ️  S3_MODEL_PATH not set — using local weights at {LOCAL_MODEL_PATH}")
    else:
        raise RuntimeError(
            "No model weights available: S3_MODEL_PATH is not set and "
            f"{LOCAL_MODEL_PATH} does not exist."
        )

    print("🚀 Loading model…")
    model = ModelLoader(LOCAL_MODEL_PATH)
    print("✅ Model loaded successfully")


# =========================
# Shared inference
# =========================
def run_inference(img: Image.Image):
    img = img.convert("RGB")

    rgb, ela = preprocess(img)

    output = model.predict(rgb, ela)

    prob_real = float(output[0][0].item())
    prob_fake = float(output[0][1].item())

    pred = 1 if prob_fake > THRESHOLD else 0

    return rgb, ela, prob_real, prob_fake, pred, img


# =========================
# Grad-CAM helper
# =========================
def run_explain(rgb, ela, orig_img):
    target_layer = model.model.rgb_features[-1]

    wrapped_model = WrapperModel(model.model, ela)

    cam = GradCAM(
        model=wrapped_model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(input_tensor=rgb)[0]

    img_resized = orig_img.resize((224, 224))
    img_np = np.array(img_resized) / 255.0

    visualization = show_cam_on_image(
        img_np.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    _, buffer = cv2.imencode(".jpg", visualization)
    explanation = base64.b64encode(buffer).decode("utf-8")

    return explanation


# =========================
# Health check
# =========================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Forgery Detection API is live"
    }


# =========================
# 1. Detect only
# =========================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)

        _, _, prob_real, prob_fake, pred, _ = run_inference(img)

        return {
            "prediction": int(pred),
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "threshold_used": THRESHOLD
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# 3. Detect + Explain
# =========================
@app.post("/detect-explain")
async def detect_explain(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)

        rgb, ela, prob_real, prob_fake, pred, orig_img = run_inference(img)

        explanation = run_explain(rgb, ela, orig_img)

        return {
            "prediction": int(pred),
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "threshold_used": THRESHOLD,
            "explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}
