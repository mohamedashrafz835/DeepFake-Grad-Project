ARG ECR_REGISTRY
ARG BASE_TAG=base-latest
FROM ${ECR_REGISTRY}/deepfake-base/api-gateway:${BASE_TAG}

WORKDIR /app

# Copy application code
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
