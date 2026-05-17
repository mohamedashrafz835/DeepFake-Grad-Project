variable "vpc_id" {
  description = "VPC ID to deploy resources into"
  type        = string
}

variable "subnet_ids" {
  description = "List of public subnet IDs (2) for EC2 and ALB"
  type        = list(string)
}

variable "key_pair_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
}

variable "ami_id" {
  description = "AMI ID for EC2 instances"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "aws_account_id" {
  description = "AWS account ID (used to build ECR URLs)"
  type        = string
}

variable "github_owner" {
  description = "GitHub owner for the project repository"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
}

variable "s3_bucket_name" {
  description = "S3 model bucket name (passed to EC2 for model downloads)"
  type        = string
}
