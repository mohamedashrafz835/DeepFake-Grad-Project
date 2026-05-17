variable "github_owner" {
  description = "GitHub username or organization that owns the repository"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
}

variable "s3_bucket_name" {
  description = "Name of the ML models S3 bucket (for IAM policy)"
  type        = string
}

variable "s3_bucket_arn" {
  description = "ARN of the ML models S3 bucket (for IAM policy)"
  type        = string
}
