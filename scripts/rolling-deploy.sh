#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Rolling Deploy Script
# Usage: ./rolling-deploy.sh <service_name> <image_tag>
#
# Examples:
#   ./rolling-deploy.sh text-service  v12
#   ./rolling-deploy.sh image-service v7
#   ./rolling-deploy.sh api-gateway   latest
#   ./rolling-deploy.sh frontend      sha-abc1234
#
# Strategy (zero-downtime):
#   1. Pull new image from ECR
#   2. Start NEW container on a temp port
#   3. Health-check the new container (retry loop)
#   4. Stop and remove OLD container
#   5. Start definitive container on the correct port
#   6. Final health check
#
# Port map:
#   frontend      → 3000
#   api-gateway   → 5000
#   text-service  → 5001
#   image-service → 5002
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────
SERVICE="${1:-}"
IMAGE_TAG="${2:-latest}"

if [[ -z "$SERVICE" ]]; then
  echo "Usage: $0 <service_name> [image_tag]"
  exit 1
fi

# ── Config ────────────────────────────────────────────────────────
APP_DIR="/opt/deepfake"
AWS_REGION="${AWS_REGION:-us-east-1}"
# ECR_REGISTRY is injected via environment or .env file
source "$APP_DIR/.env" 2>/dev/null || true

# Map service → production port
declare -A PORT_MAP=(
  [frontend]=3000
  [api-gateway]=5000
  [text-service]=5001
  [image-service]=5002
)

# Map service → health check path
declare -A HEALTH_PATH=(
  [frontend]="/"
  [api-gateway]="/health"
  [text-service]="/health"
  [image-service]="/health"
)

# ── Validate service ──────────────────────────────────────────────
if [[ -z "${PORT_MAP[$SERVICE]+_}" ]]; then
  echo "ERROR: Unknown service '$SERVICE'"
  echo "Valid services: ${!PORT_MAP[*]}"
  exit 1
fi

PROD_PORT="${PORT_MAP[$SERVICE]}"
TEMP_PORT=$(( PROD_PORT + 10000 ))   # e.g. text-service temp port = 15001
HEALTH="${HEALTH_PATH[$SERVICE]}"
IMAGE="${ECR_REGISTRY}/deepfake/${SERVICE}:${IMAGE_TAG}"
CONTAINER_NAME="deepfake-${SERVICE}"
TEMP_CONTAINER="deepfake-${SERVICE}-new"
NETWORK="deepfake-net"

echo "═══════════════════════════════════════════════"
echo " Rolling deploy: $SERVICE → $IMAGE"
echo " Prod port: $PROD_PORT | Temp port: $TEMP_PORT"
echo "═══════════════════════════════════════════════"

# ── Step 1: Authenticate with ECR ────────────────────────────────
echo "[1/6] Authenticating with ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# ── Step 2: Pull new image ────────────────────────────────────────
echo "[2/6] Pulling $IMAGE..."
docker pull "$IMAGE"

# ── Step 3: Start new container on temp port ──────────────────────
echo "[3/6] Starting new container on temp port $TEMP_PORT..."
docker rm -f "$TEMP_CONTAINER" 2>/dev/null || true
docker run -d \
  --name "$TEMP_CONTAINER" \
  --network "$NETWORK" \
  -p "${TEMP_PORT}:${PROD_PORT}" \
  --restart no \
  "$IMAGE"

# ── Step 4: Health-check the new container ────────────────────────
echo "[4/6] Health-checking new container..."
MAX_RETRIES=12
RETRY_INTERVAL=5
HEALTHY=false

for i in $(seq 1 $MAX_RETRIES); do
  echo "  Attempt $i/$MAX_RETRIES → GET http://localhost:${TEMP_PORT}${HEALTH}"
  HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 5 \
    "http://localhost:${TEMP_PORT}${HEALTH}" 2>/dev/null || echo "000")

  if [[ "$HTTP_STATUS" == "200" ]]; then
    echo "  ✓ Health check passed (HTTP $HTTP_STATUS)"
    HEALTHY=true
    break
  fi
  echo "  ✗ HTTP $HTTP_STATUS — waiting ${RETRY_INTERVAL}s..."
  sleep "$RETRY_INTERVAL"
done

if [[ "$HEALTHY" != "true" ]]; then
  echo "ERROR: New container failed health checks after $MAX_RETRIES attempts."
  echo "Rolling back — removing failed container..."
  docker rm -f "$TEMP_CONTAINER"
  exit 1
fi

# ── Step 5: Remove old container ──────────────────────────────────
echo "[5/6] Stopping old container '$CONTAINER_NAME'..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null \
  && echo "  ✓ Old container removed" \
  || echo "  (No old container found — first deploy)"

# Remove the temp container too (we'll recreate on the real port)
docker rm -f "$TEMP_CONTAINER" 2>/dev/null || true

# ── Step 6: Start definitive container on real port ───────────────
echo "[6/6] Starting '$CONTAINER_NAME' on port $PROD_PORT..."

# Build service-specific run args
EXTRA_ARGS=""
if [[ "$SERVICE" == "api-gateway" ]]; then
  EXTRA_ARGS="-e TEXT_SERVICE_URL=http://text-service:5001 -e IMAGE_SERVICE_URL=http://image-service:5002"
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --network "$NETWORK" \
  -p "${PROD_PORT}:${PROD_PORT}" \
  --restart unless-stopped \
  $EXTRA_ARGS \
  "$IMAGE"

# Final health check on real port
echo "  Final health check on port $PROD_PORT..."
sleep 5
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  --max-time 10 \
  "http://localhost:${PROD_PORT}${HEALTH}" 2>/dev/null || echo "000")

if [[ "$HTTP_STATUS" == "200" ]]; then
  echo ""
  echo "✅ Deploy successful: $SERVICE@$IMAGE_TAG is live on port $PROD_PORT"
else
  echo "WARNING: Final health check returned HTTP $HTTP_STATUS"
  echo "Container may still be starting up — check logs:"
  echo "  docker logs $CONTAINER_NAME"
  exit 1
fi
