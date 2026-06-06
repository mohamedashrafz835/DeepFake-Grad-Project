# ===================================================
# Stage 1: Builder
# ===================================================
FROM python:3.10-slim AS builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first
COPY requirements.txt .

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU FIRST (fixed & stable)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (NO DEPS to avoid conflicts)
RUN pip install --no-cache-dir --no-deps -r requirements.txt


# ===================================================
# Stage 2: Runner
# ===================================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Runtime dependencies (IMPORTANT for cv2 + torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv
COPY --from=builder /opt/venv /opt/venv

# Copy app
COPY app/ ./app/
COPY start.sh ./start.sh

RUN chmod +x start.sh

# Activate venv
ENV PATH="/opt/venv/bin:$PATH"

# Python config
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Performance tuning
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 8000

CMD ["./start.sh"]