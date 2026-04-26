from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import httpx
import asyncio

app = FastAPI()

TEXT_SERVICE_URL = "http://text-service:8000/predict"
IMAGE_SERVICE_URL = "http://image-service:8000/predict"

@app.post("/detect")
async def detect(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    async with httpx.AsyncClient() as client:

        tasks = []

        # لو فيه text
        if text:
            tasks.append(
                client.post(TEXT_SERVICE_URL, params={"text": text})
            )

        # لو فيه image
        if image:
            image_bytes = await image.read()
            tasks.append(
                client.post(
                    IMAGE_SERVICE_URL,
                    files={"file": ("image.jpg", image_bytes)}
                )
            )

        # لو مفيش ولا واحد
        if not tasks:
            return {"error": "Please provide text or image"}

        responses = await asyncio.gather(*tasks)

    result = {}

    idx = 0
    if text:
        result["text_result"] = responses[idx].json()
        idx += 1

    if image:
        result["image_result"] = responses[idx].json()

    return result
