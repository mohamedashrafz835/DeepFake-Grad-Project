FROM python:3.10-slim

WORKDIR /app

# Copy only requirements to cache them in the base image
# Context is assumed to be project root
COPY api-gateway/requirements.txt .

# Install python dependencies + boto3/awscli for runtime model loading from S3
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt boto3 awscli
