output "alb_dns_name" {
  description = "Public DNS of the Application Load Balancer — open this in your browser"
  value       = module.ec2.alb_dns_name
}

output "ec2_instance_ids" {
  description = "IDs of both EC2 instances"
  value       = module.ec2.ec2_instance_ids
}

output "ec2_public_ips" {
  description = "Public IPs of both EC2 instances (for SSH)"
  value       = module.ec2.ec2_public_ips
}

output "model_bucket_name" {
  description = "S3 bucket name for ML model weights"
  value       = module.s3.bucket_name
}

output "lambda_function_name" {
  description = "Name of the MLOps router Lambda function"
  value       = module.lambda.lambda_function_name
}

output "ecr_repos" {
  description = "ECR repository URLs for all 4 services"
  value       = module.ec2.ecr_repo_urls
}
