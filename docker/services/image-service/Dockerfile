ARG ECR_REGISTRY
ARG BASE_TAG=base-latest
FROM ${ECR_REGISTRY}/deepfake-base/image-service:${BASE_TAG}

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Runtime configuration for model loading
ENV MODEL_CONFIG_S3_KEY="image-service/current-model.json"

CMD ["./start.sh"]
