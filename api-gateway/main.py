from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import httpx
import asyncio

app = FastAPI(title="API Gateway")

# =========================
# Services URLs
# =========================
TEXT_SERVICE_URL = "http://text-service:8000/predict"
TEXT_SERVICE_EXPLAIN_URL = "http://text-service:8000/predict_with_explanation"

IMAGE_SERVICE_URL = "http://image-service:8000/detect"
IMAGE_SERVICE_EXPLAIN_URL = "http://image-service:8000/detect-explain"


# =========================
# Health check
# =========================
@app.get("/")
def home():
    return {"message": "API Gateway is running 🚀"}


# =========================
# Main detect endpoint
# =========================
@app.post("/detect")
async def detect(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),

    text_explain: bool = Form(False),
    image_explain: bool = Form(False)
):
    async with httpx.AsyncClient(timeout=120.0) as client:

        tasks = []
        responses_map = []

        # =====================
        # TEXT SERVICE
        # =====================
        if text:
            url = TEXT_SERVICE_EXPLAIN_URL if text_explain else TEXT_SERVICE_URL

            tasks.append(
                client.post(url, params={"text": text})
            )
            responses_map.append("text")

        # =====================
        # IMAGE SERVICE (FIXED)
        # =====================
        if image:
            image_bytes = await image.read()

            url = IMAGE_SERVICE_EXPLAIN_URL if image_explain else IMAGE_SERVICE_URL

            files = {
                "file": (
                    image.filename or "image.jpg",
                    image_bytes,
                    image.content_type or "image/jpeg"
                )
            }

            tasks.append(
                client.post(url, files=files)
            )
            responses_map.append("image")

        # =====================
        # Validate input
        # =====================
        if not tasks:
            return {"error": "Please provide text or image"}

        # =====================
        # Execute in parallel
        # =====================
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    # =====================
    # Build response safely
    # =====================
    result = {}

    for i, key in enumerate(responses_map):

        resp = responses[i]

        # handle errors safely
        if isinstance(resp, Exception):
            result[f"{key}_error"] = str(resp)
            continue

        try:
            result[f"{key}_result"] = resp.json()
        except Exception:
            result[f"{key}_error"] = resp.text

    return result
