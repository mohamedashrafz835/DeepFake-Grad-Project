variable "bucket_name" {
  description = "Name of the S3 bucket for ML model weights"
  type        = string
}

variable "lambda_arn" {
  description = "ARN of the Lambda function to notify on new model uploads"
  type        = string
}
