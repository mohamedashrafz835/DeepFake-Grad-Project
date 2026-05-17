ARG ECR_REGISTRY
ARG BASE_TAG=base-latest
FROM ${ECR_REGISTRY}/deepfake-base/text-service:${BASE_TAG}

WORKDIR /app

# Copy application code
COPY app/ ./app/

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV MODEL_CONFIG_S3_KEY="text-service/current-model.json"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
