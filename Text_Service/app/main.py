import os
import torch
import gdown

from fastapi import FastAPI
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer

from app.model import AraBertCNNLSTMClassifier
from app.utils import arabert_preprocess

app = FastAPI()

MODEL_PATH = "/app/model.pt"
DRIVE_URL = "https://drive.google.com/uc?id=1CmBumK7XU8DkMP4mvFeXo7OnPAj5wV1y"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
tokenizer = None
explainer = None


# =========================
# Startup
# =========================
@app.on_event("startup")
def load_everything():
    global model, tokenizer, explainer

    print(f"🔥 Startup PID: {os.getpid()}")

    s3_uri = os.environ.get("MODEL_CONFIG_S3_KEY")
    
    if s3_uri and s3_uri.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse
        
        tmp_path = MODEL_PATH + ".tmp"
        print(f"⬇️ Downloading model from {s3_uri}...")
        
        s3 = boto3.client('s3')
        parsed = urlparse(s3_uri)
        s3.download_file(parsed.netloc, parsed.path.lstrip('/'), tmp_path)
        os.rename(tmp_path, MODEL_PATH)
        print("✅ Download complete.")
        
    # fallback to gdown if no S3 URI provided and model doesn't exist
    elif not os.path.exists(MODEL_PATH):
        tmp_path = MODEL_PATH + ".tmp"

        if not os.path.exists(tmp_path):
            print("⬇️ Downloading model via gdown (fallback)...")
            gdown.download(DRIVE_URL, tmp_path, quiet=False)

        os.rename(tmp_path, MODEL_PATH)

    print("🧠 Loading model...")

    model = AraBertCNNLSTMClassifier()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "aubmindlab/bert-base-arabertv02"
    )

    explainer = LimeTextExplainer(
        class_names=["credible", "not credible"]
    )

    print("✅ Model + LIME ready!")


# =========================
# Prediction helper
# =========================
def predict_proba(texts):
    global model, tokenizer

    texts = [arabert_preprocess(t) for t in texts]

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=384,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device)
        )

        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


# =========================
# Format LIME output
# =========================
def format_explanation(explanation):
    if not explanation:
        return []

    max_abs = max(abs(score) for _, score in explanation)

    formatted = []

    for word, score in explanation:
        formatted.append({
            "word": word,
            "impact": float(score),
            "normalized_impact": float(abs(score) / max_abs if max_abs != 0 else 0),
            "color": "green" if score > 0 else "red"
        })

    formatted.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return formatted


# =========================
# Natural explanation (Arabic)
# =========================
def generate_natural_explanation(explanation, prediction):
    if not explanation:
        return ""

    positives = [e["word"] for e in explanation if e["impact"] > 0]
    negatives = [e["word"] for e in explanation if e["impact"] < 0]

    text = ""

    if prediction == "credible":
        text += "النموذج توقع أن النص موثوق (credible) لأن:\n"

        if positives:
            text += f"- الكلمات التي دعمت القرار: {', '.join(positives[:5])}\n"

        if negatives:
            text += f"- لكن بعض الكلمات خففت الثقة: {', '.join(negatives[:5])}\n"

    else:
        text += "النموذج توقع أن النص غير موثوق (not credible) لأن:\n"

        if negatives:
            text += f"- الكلمات التي دعمت القرار: {', '.join(negatives[:5])}\n"

        if positives:
            text += f"- لكن بعض الكلمات عارضت القرار: {', '.join(positives[:5])}\n"

    return text


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "Model is running 🚀"}


@app.post("/predict")
def predict(text: str):
    global model, tokenizer

    text_clean = arabert_preprocess(text)

    enc = tokenizer(
        text_clean,
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

    label = "credible" if pred == 0 else "not credible"

    return {
        "prediction": label,
        "confidence": float(probs[0][pred])
    }


# =========================
# LIME endpoint
# =========================
@app.post("/predict_with_explanation")
def predict_with_explanation(text: str):
    global explainer

    # prediction
    pred_result = predict(text)

    # LIME
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=10,
        num_samples=20
    )

    raw = exp.as_list()
    explanation = format_explanation(raw)

    natural_text = generate_natural_explanation(
        explanation,
        pred_result["prediction"]
    )

    return {
        **pred_result,
        "explanation": explanation,
        "explanation_text": natural_text
    }
