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

# =========================
# FIXED threshold (temporary)
# (better: calibrate later)
# =========================
THRESHOLD = 0.384


# =========================
# WRAPPER FOR GRAD-CAM
# =========================
class WrapperModel(torch.nn.Module):
    def __init__(self, model, ela):
        super().__init__()
        self.model = model
        self.ela = ela

    def forward(self, x):
        return self.model(x, self.ela)


# =========================
# LOAD MODEL
# =========================
@app.on_event("startup")
def load_model():
    global model
    print("🚀 Loading model...")
    model = ModelLoader("/app/model.pth")
    print("✅ Model loaded")


# =========================
# INFERENCE
# =========================
def run_inference(img: Image.Image):
    rgb, ela = preprocess(img)

    probs = model.predict(rgb, ela)

    prob_real = float(probs[0, 0])
    prob_fake = float(probs[0, 1])

    pred = int(prob_fake > THRESHOLD)

    return rgb, ela, prob_real, prob_fake, pred, img


# =========================
# GRAD-CAM
# =========================
def run_explain(rgb, ela, orig_img):
    target_layer = model.model.rgb_features[-1]

    wrapped = WrapperModel(model.model, ela)

    cam = GradCAM(model=wrapped, target_layers=[target_layer])

    grayscale = cam(input_tensor=rgb)[0]

    img_resized = orig_img.resize((224, 224))
    img_np = np.array(img_resized) / 255.0

    viz = show_cam_on_image(
        img_np.astype(np.float32),
        grayscale,
        use_rgb=True
    )

    _, buffer = cv2.imencode(".jpg", viz)
    return base64.b64encode(buffer).decode()


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Forgery API running"}


# =========================
# DETECT
# =========================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)

        _, _, prob_real, prob_fake, pred, _ = run_inference(img)

        return {
            "prediction": pred,
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "threshold": THRESHOLD
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# DETECT + EXPLAIN
# =========================
@app.post("/detect-explain")
async def detect_explain(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)

        rgb, ela, prob_real, prob_fake, pred, orig = run_inference(img)

        explanation = run_explain(rgb, ela, orig)

        return {
            "prediction": pred,
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "threshold": THRESHOLD,
            "explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}
