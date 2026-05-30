# ===================================================
# Stage 1: Builder
# ===================================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Build dependencies only (removed from final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first (better caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU first (important for ML stack stability)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt


# ===================================================
# Stage 2: Runner
# ===================================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Runtime system dependencies (required for torch, cv2, grad-cam)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app/ ./app/
COPY start.sh ./start.sh

# Make start script executable
RUN chmod +x start.sh

# Activate venv
ENV PATH="/opt/venv/bin:$PATH"

# Fix Python imports
ENV PYTHONPATH=/app

# Performance tuning for ML inference
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run via start script (handles model download + API start)
CMD ["./start.sh"]