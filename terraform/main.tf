# ─────────────────────────────────────────────────────────────────
# Root Module — wires all child modules together
# ─────────────────────────────────────────────────────────────────

# ── VPC ──────────────────────────────────────────────────────────
module "vpc" {
  source = "./vpc"

  aws_region          = var.aws_region
  vpc_cidr            = "10.0.0.0/16"
  public_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24"]
  azs                 = ["${var.aws_region}a", "${var.aws_region}b"]
}

# ── Lambda (must come before S3 so we have the ARN) ──────────────
module "lambda" {
  source = "./lambda"

  github_owner   = var.github_owner
  github_repo    = var.github_repo
  s3_bucket_name = module.s3.bucket_name
  s3_bucket_arn  = module.s3.bucket_arn
}

# ── S3 Model Bucket ───────────────────────────────────────────────
module "s3" {
  source = "./s3"

  bucket_name = "ml-models-deepfake-${data.aws_caller_identity.current.account_id}"
  lambda_arn  = module.lambda.lambda_arn
}

# ── EC2 + ALB ────────────────────────────────────────────────────
module "ec2" {
  source = "./ec2"

  vpc_id        = module.vpc.vpc_id
  subnet_ids    = module.vpc.public_subnet_ids
  key_pair_name = var.key_pair_name
  ami_id        = var.ami_id
  instance_type = var.instance_type

  # Pass ECR registry URL so user_data can authenticate
  aws_region      = var.aws_region
  aws_account_id  = data.aws_caller_identity.current.account_id
  github_owner    = var.github_owner
  github_repo     = var.github_repo
  s3_bucket_name  = module.s3.bucket_name
}

# ── Data sources ─────────────────────────────────────────────────
data "aws_caller_identity" "current" {}
