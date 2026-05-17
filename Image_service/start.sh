#!/bin/bash

if [[ -n "$MODEL_CONFIG_S3_KEY" && "$MODEL_CONFIG_S3_KEY" == s3://* ]]; then
    echo "⬇️ Downloading model from S3: $MODEL_CONFIG_S3_KEY"
    aws s3 cp "$MODEL_CONFIG_S3_KEY" /app/model.pth
    echo "✅ S3 Download complete."
elif [ ! -f "/app/model.pth" ]; then
    echo "⬇️ Downloading model via gdown (fallback)..."
    gdown "https://drive.google.com/uc?id=14Qskqu6-p4qhRv_4hekFXuIwVyxtkJhJ" -O /app/model.pth
    echo "✅ Fallback download complete."
else
    echo "✅ Model already exists locally."
fi
echo "🚀 Starting API..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
