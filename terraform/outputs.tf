
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "model_bucket_name" {
  description = "S3 bucket name for ML model weights"
  value       = module.s3.bucket_name
}

output "lambda_function_name" {
  description = "Name of the MLOps router Lambda function"
  value       = module.lambda.lambda_function_name
}
