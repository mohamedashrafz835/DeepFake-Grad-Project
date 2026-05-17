#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# EC2 User Data Bootstrap Script
# Runs ONCE on first boot of each EC2 instance.
# Installs Docker, Docker Compose, AWS CLI, pulls images from ECR
# and starts all 4 containers via docker-compose.
#
# Template variables (replaced by Terraform templatefile):
#   ${aws_region}     — e.g. us-east-1
#   ${aws_account_id} — 12-digit AWS account ID
#   ${ecr_registry}   — <account>.dkr.ecr.<region>.amazonaws.com
#   ${github_owner}   — GitHub username/org
#   ${github_repo}    — GitHub repo name
#   ${s3_bucket_name} — ML model S3 bucket
# ─────────────────────────────────────────────────────────────────

set -euxo pipefail
exec > /var/log/user-data.log 2>&1

# ── 1. System update ──────────────────────────────────────────────
yum update -y

# ── 2. Install Docker ─────────────────────────────────────────────
yum install -y docker git
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# ── 3. Install Docker Compose v2 ──────────────────────────────────
COMPOSE_VERSION="2.27.0"
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
# Also create legacy alias
ln -sf /usr/local/lib/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose

# ── 4. Install AWS CLI v2 ─────────────────────────────────────────
yum install -y unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

# ── 5. Install CloudWatch agent ───────────────────────────────────
yum install -y amazon-cloudwatch-agent

# ── 6. Clone the repository ───────────────────────────────────────
APP_DIR="/opt/deepfake"
mkdir -p "$APP_DIR"
git clone "https://github.com/${github_owner}/${github_repo}.git" "$APP_DIR"
chown -R ec2-user:ec2-user "$APP_DIR"

# ── 7. Write environment file for docker-compose ─────────────────
cat > "$APP_DIR/.env" <<EOF
AWS_REGION=${aws_region}
ECR_REGISTRY=${ecr_registry}
S3_BUCKET=${s3_bucket_name}
EOF

# ── 8. Authenticate Docker with ECR ──────────────────────────────
aws ecr get-login-password --region "${aws_region}" \
  | docker login --username AWS --password-stdin "${ecr_registry}"

# ── 9. Write the production docker-compose override ───────────────
# Uses ECR images instead of local builds
cat > "$APP_DIR/docker-compose.prod.yml" <<'COMPOSE'
version: "3.9"

services:
  frontend:
    image: ${ECR_REGISTRY}/deepfake/frontend:latest
    container_name: deepfake-frontend
    ports:
      - "3000:80"
    depends_on:
      - api-gateway
    networks:
      - deepfake-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s

  api-gateway:
    image: ${ECR_REGISTRY}/deepfake/api-gateway:latest
    container_name: deepfake-api-gateway
    ports:
      - "5000:5000"
    environment:
      - TEXT_SERVICE_URL=http://text-service:5001
      - IMAGE_SERVICE_URL=http://image-service:5002
    depends_on:
      - text-service
      - image-service
    networks:
      - deepfake-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s

  text-service:
    image: ${ECR_REGISTRY}/deepfake/text-service:latest
    container_name: deepfake-text-service
    ports:
      - "5001:5001"
    networks:
      - deepfake-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  image-service:
    image: ${ECR_REGISTRY}/deepfake/image-service:latest
    container_name: deepfake-image-service
    ports:
      - "5002:5002"
    networks:
      - deepfake-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

networks:
  deepfake-net:
    driver: bridge
COMPOSE

# Replace literal ${ECR_REGISTRY} with actual value in compose file
sed -i "s|\${ECR_REGISTRY}|${ecr_registry}|g" "$APP_DIR/docker-compose.prod.yml"

# ── 10. Pull images and start services ───────────────────────────
cd "$APP_DIR"
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# ── 11. Configure CloudWatch log agent ───────────────────────────
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<CWCONFIG
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/user-data.log",
            "log_group_name": "/deepfake/ec2/user-data",
            "log_stream_name": "{instance_id}"
          },
          {
            "file_path": "/var/lib/docker/containers/**/*.log",
            "log_group_name": "/deepfake/docker",
            "log_stream_name": "{instance_id}/{hostname}"
          }
        ]
      }
    }
  }
}
CWCONFIG

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

echo "Bootstrap complete — all containers started."
