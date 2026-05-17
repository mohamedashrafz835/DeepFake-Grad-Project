output "bucket_name" {
  description = "Name of the ML models S3 bucket"
  value       = aws_s3_bucket.models.bucket
}

output "bucket_arn" {
  description = "ARN of the ML models S3 bucket"
  value       = aws_s3_bucket.models.arn
}
