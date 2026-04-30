from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import httpx
import asyncio

app = FastAPI()

TEXT_SERVICE_URL = "http://text-service:8000/predict"
TEXT_SERVICE_EXPLAIN_URL = "http://text-service:8000/predict_with_explanation"

IMAGE_SERVICE_URL = "http://image-service:8000/predict"


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
    explain: bool = Form(False)
):
    async with httpx.AsyncClient(timeout=60.0) as client:

        tasks = []

        # =====================
        # TEXT SERVICE
        # =====================
        if text:
            url = TEXT_SERVICE_EXPLAIN_URL if explain else TEXT_SERVICE_URL

            tasks.append(
                client.post(
                    url,
                    params={"text": text}
                )
            )

        # =====================
        # IMAGE SERVICE
        # =====================
        if image:
            image_bytes = await image.read()

            tasks.append(
                client.post(
                    IMAGE_SERVICE_URL,
                    files={"file": ("image.jpg", image_bytes)}
                )
            )

        # لو مفيش input
        if not tasks:
            return {
                "error": "Please provide text or image"
            }

        # =====================
        # Execute in parallel
        # =====================
        responses = await asyncio.gather(*tasks)

    # =====================
    # Build response
    # =====================
    result = {}
    idx = 0

    if text:
        result["text_result"] = responses[idx].json()
        idx += 1

    if image:
        result["image_result"] = responses[idx].json()

    return result
