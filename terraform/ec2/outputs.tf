output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "ec2_instance_ids" {
  description = "EC2 instance IDs"
  value       = aws_instance.app[*].id
}

output "ec2_public_ips" {
  description = "Public IP addresses of both EC2 instances"
  value       = aws_instance.app[*].public_ip
}

output "ecr_repo_urls" {
  description = "ECR repository URLs keyed by service name"
  value       = { for k, v in aws_ecr_repository.service : k => v.repository_url }
}
