FROM python:3.10-slim

WORKDIR /app

# System dependencies (important for OpenCV and PIL)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to cache them in the base image
# Context is assumed to be project root
COPY Image_service/requirements.txt .

# Install python dependencies + boto3/awscli for runtime model loading from S3
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt boto3 awscli
