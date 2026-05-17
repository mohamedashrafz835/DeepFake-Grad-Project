terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }

  backend "s3" {
    bucket         = "statefile-bucket-2210xz"
    key            = "global/s3/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "statefile-table"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "deepfake-detector"
      Environment = "production"
      ManagedBy   = "terraform"
    }
  }
}
