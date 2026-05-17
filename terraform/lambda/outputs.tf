output "lambda_arn" {
  description = "ARN of the MLOps router Lambda function"
  value       = aws_lambda_function.mlops_router.arn
}

output "lambda_function_name" {
  description = "Name of the MLOps router Lambda function"
  value       = aws_lambda_function.mlops_router.function_name
}

output "github_pat_secret_name" {
  description = "Secrets Manager secret name — populate this with your GitHub PAT after apply"
  value       = aws_secretsmanager_secret.github_pat.name
}
