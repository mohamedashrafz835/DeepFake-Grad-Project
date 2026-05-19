# ─────────────────────────────────────────────────────────────────
# S3 Module — ML Model Weights Bucket
#
# Bucket layout:
#   s3://ml-models-<account-id>/
#     ├── text-service/v1/model.pt   → triggers text-service-deploy.yml
#     └── image-service/v1/model.pt  → triggers image-service-deploy.yml
#
# Features:
#   - Versioning (never lose a model weight)
#   - AES-256 server-side encryption
#   - No public access
#   - S3 event notifications → Lambda (per-prefix filtering)
#   - Lifecycle rule: expire non-current versions after 30 days
# ─────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "models" {
  bucket = var.bucket_name

  tags = {
    Name    = "ml-models"
    Purpose = "model-weights"
  }
}

# ── Versioning ────────────────────────────────────────────────────
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ── Server-side encryption ────────────────────────────────────────
resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# ── Block all public access ───────────────────────────────────────
resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── Lifecycle: expire old non-current model versions ─────────────
resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  # text-service — keep last 30 days of non-current versions
  rule {
    id     = "expire-old-text-service-models"
    status = "Enabled"

    filter {
      prefix = "text-service/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  # image-service — keep last 30 days of non-current versions
  rule {
    id     = "expire-old-image-service-models"
    status = "Enabled"

    filter {
      prefix = "image-service/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# ── S3 Event Notifications → Lambda ──────────────────────────────
# Both prefixes notify the SAME Lambda function.
# The Lambda's routing logic distinguishes which service to deploy.
resource "aws_s3_bucket_notification" "model_upload" {
  bucket = aws_s3_bucket.models.id

  lambda_function {
    lambda_function_arn = var.lambda_arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "text-service/"
    filter_suffix       = ".pt"
  }

  lambda_function {
    lambda_function_arn = var.lambda_arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "image-service/"
    filter_suffix       = ".pth"
  }

  depends_on = [aws_s3_bucket.models]
}
