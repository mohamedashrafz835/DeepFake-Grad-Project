# ─────────────────────────────────────────────────────────────────
# Lambda Module — MLOps Router
#
# Resources:
#   - IAM role + least-privilege policy
#   - Secrets Manager secret (GitHub PAT)
#   - Lambda function (Python 3.12, zipped from src/)
#   - Lambda permission (allows S3 to invoke it)
# ─────────────────────────────────────────────────────────────────

# ── Package the Lambda source code ────────────────────────────────
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/src"
  output_path = "${path.module}/lambda_package.zip"
}

# ── IAM Role for Lambda ───────────────────────────────────────────
resource "aws_iam_role" "lambda_role" {
  name = "deepfake-mlops-router-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "deepfake-mlops-router-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # CloudWatch Logs
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      # S3: read the uploaded model object metadata
      {
        Sid    = "S3ModelRead"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectTagging"
        ]
        Resource = "${var.s3_bucket_arn}/*"
      },
      # Secrets Manager: fetch GitHub PAT at runtime
      {
        Sid    = "GetGitHubPAT"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.github_pat.arn
      }
    ]
  })
}

# ── GitHub PAT Secret ─────────────────────────────────────────────
# After terraform apply, populate the value with:
#   aws secretsmanager put-secret-value \
#     --secret-id deepfake/github-pat \
#     --secret-string '{"token":"ghp_YOUR_TOKEN_HERE"}'
resource "aws_secretsmanager_secret" "github_pat" {
  name                    = "deepfake/github-pat"
  description             = "GitHub Personal Access Token for workflow_dispatch (scope: actions:write)"
  recovery_window_in_days = 0  # allow immediate deletion during development
}

# ── Lambda Function ───────────────────────────────────────────────
resource "aws_lambda_function" "mlops_router" {
  function_name    = "deepfake-mlops-router"
  role             = aws_iam_role.lambda_role.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.12"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = 30
  memory_size      = 128

  environment {
    variables = {
      GITHUB_OWNER      = var.github_owner
      GITHUB_REPO       = var.github_repo
      GITHUB_SECRET_ARN = aws_secretsmanager_secret.github_pat.arn
    }
  }

  depends_on = [
    aws_iam_role_policy.lambda_policy,
    data.archive_file.lambda_zip
  ]
}

# ── CloudWatch Log Group for Lambda ──────────────────────────────
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.mlops_router.function_name}"
  retention_in_days = 14
}

# ── Allow S3 to invoke the Lambda ─────────────────────────────────
resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.mlops_router.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = var.s3_bucket_arn
}
