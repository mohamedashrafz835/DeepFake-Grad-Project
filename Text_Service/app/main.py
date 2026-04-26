import os
import torch
import gdown
from fastapi import FastAPI
from transformers import AutoTokenizer

from app.model import AraBertCNNLSTMClassifier
from app.utils import arabert_preprocess

app = FastAPI()

MODEL_PATH = "/app/model.pt"
DRIVE_URL = "https://drive.google.com/file/d/1CmBumK7XU8DkMP4mvFeXo7OnPAj5wV1y/view?usp=drive_link"  # 👈 replace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Download model (ONCE)
# ---------------------------
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)

# ---------------------------
# Load model (ONCE)
# ---------------------------
print("🧠 Loading model...")

model = AraBertCNNLSTMClassifier()
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("✅ Model loaded!")

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return {"message": "Model is running 🚀"}

@app.post("/predict")
def predict(text: str):
    text = arabert_preprocess(text)

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=384,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device)
        )
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return {
        "prediction": "credible" if pred == 0 else "not credible",
        "confidence": float(probs[0][pred])
    }
