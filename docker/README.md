# Deepfake Detection — Layered Docker Architecture

This repository uses a two-layer Docker architecture to dramatically speed up CI/CD build times and minimize deployment sizes.

## How It Works

### 1. Base Images (`docker/base/`)
Base images are heavy, monolithic layers that contain all system dependencies (CUDA, OpenCV, PyTorch) and Python packages (`requirements.txt`).
- **When they build**: ONLY when you change `requirements.txt` or a `docker/base/**/Dockerfile`.
- **Why**: Installing PyTorch and OpenCV takes minutes. By caching this in a base image, your daily code deployments skip this entirely.
- **Where they live**: ECR repositories named `deepfake-base/<service>`

### 2. Service Images (`docker/services/`)
Service images are lightweight layers that extend the base images (`FROM deepfake-base/...`). They only copy the actual Python/HTML application code.
- **When they build**: On every `push` to `main` modifying source code.
- **Why**: Copying a few Python scripts takes seconds. Your CI/CD pipeline becomes extremely fast.
- **Where they live**: ECR repositories named `deepfake/<service>`

### 3. Runtime Model Loading
Models are **NOT** baked into the Docker images. 
Instead, the service containers read an environment variable `MODEL_CONFIG_S3_KEY` at startup, fetch the configuration, and download the current weights from S3 dynamically.
- **When models update**: S3 upload → triggers Lambda → calls `workflow_dispatch` on the GitHub Action deploy workflows.
- **Why**: Zero Docker rebuilds for model tuning/updates. Fast rolling deployments.

---

## ECR Repository Setup

Before using this architecture, ensure all 8 ECR repositories exist in your AWS account.

**Base Repositories:**
```bash
aws ecr create-repository --repository-name deepfake-base/image-service
aws ecr create-repository --repository-name deepfake-base/text-service
aws ecr create-repository --repository-name deepfake-base/api-gateway
aws ecr create-repository --repository-name deepfake-base/frontend
```

**Service Repositories (Existing):**
```bash
aws ecr create-repository --repository-name deepfake/image-service
aws ecr create-repository --repository-name deepfake/text-service
aws ecr create-repository --repository-name deepfake/api-gateway
aws ecr create-repository --repository-name deepfake/frontend
```

---

## Local Development

You can still use `docker-compose` locally! Just ensure your `.env` file is set up correctly.

### 1. Setup `.env`
Create a `.env` file in the root directory:
```env
ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
TAG=latest
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET
```

### 2. Login to ECR
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
```

### 3. Run Locally
Docker Compose is configured to pull the pre-built service images from ECR rather than building locally.
```bash
docker-compose --env-file .env up -d
```

---

## Manual Build Commands

If you need to manually build and push outside of GitHub Actions:

### Building Base Images (Example: image-service)
```bash
export ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com

docker build \
  -f docker/base/image-service/Dockerfile \
  -t $ECR_REGISTRY/deepfake-base/image-service:base-latest \
  .

docker push $ECR_REGISTRY/deepfake-base/image-service:base-latest
```

### Building Service Images (Example: image-service)
```bash
export ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com

docker build \
  -f docker/services/image-service/Dockerfile \
  --build-arg ECR_REGISTRY=$ECR_REGISTRY \
  --build-arg BASE_TAG=base-latest \
  -t $ECR_REGISTRY/deepfake/image-service:latest \
  ./Image_service

docker push $ECR_REGISTRY/deepfake/image-service:latest
```
