#!/bin/bash

echo "Downloading model..."

gdown "https://drive.google.com/uc?id=14Qskqu6-p4qhRv_4hekFXuIwVyxtkJhJ" -O /app/model.pth
echo "Model downloaded."
echo "🚀 Starting API..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
