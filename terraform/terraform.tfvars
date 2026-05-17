# ─────────────────────────────────────────────────────────────────
# FILL IN YOUR VALUES BEFORE RUNNING terraform apply
# ─────────────────────────────────────────────────────────────────

aws_region    = "us-east-1"

# The name of an existing EC2 key pair in your AWS account
# Create one via: AWS Console → EC2 → Key Pairs → Create key pair
key_pair_name = "your-key-pair-name"

# GitHub identity
github_owner  = "your-github-username"
github_repo   = "DeepFake-Grad-Project"

# t3.medium = 2 vCPU / 4 GB RAM — suitable for running all 4 containers
instance_type = "t3.medium"

# Amazon Linux 2023 (us-east-1) — verify at:
# https://aws.amazon.com/amazon-linux-ami/
ami_id = "ami-0c02fb55956c7d316"
