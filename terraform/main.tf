# ─────────────────────────────────────────────────────────────────
# Root Module — wires all child modules together
# ─────────────────────────────────────────────────────────────────

# Needed to get the AWS account ID for the S3 bucket name
data "aws_caller_identity" "current" {}

# ── VPC ──────────────────────────────────────────────────────────
module "vpc" {
  source = "./vpc"

  aws_region          = var.aws_region
  vpc_cidr            = "10.0.0.0/16"
  public_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24"]
  azs                 = ["${var.aws_region}a", "${var.aws_region}b"]
  cluster_name        = var.cluster_name
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

module "eks" {
  source = "./eks"
  cluster_name = var.cluster_name
  cluster_version = var.cluster_version
  subnets_id = module.vpc.public_subnet_ids
  vpc_id = module.vpc.vpc_id
  node_groups = var.node_groups
}
