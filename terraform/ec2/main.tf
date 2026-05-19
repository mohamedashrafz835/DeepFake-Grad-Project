# ─────────────────────────────────────────────────────────────────
# EC2 Module
# Creates: ECR repos, Security Groups, IAM role, 2x EC2, ALB,
#          Target Groups, Listeners, Path-based routing rules
# ─────────────────────────────────────────────────────────────────

locals {
  # ECR repos exist for all 4 services; ALB only exposes frontend
  services = ["frontend", "api-gateway", "text-service", "image-service"]
  ecr_registry = "${var.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

# ── ECR Repositories (one per service) ───────────────────────────
resource "aws_ecr_repository" "service" {
  for_each             = toset(local.services)
  name                 = "deepfake/${each.key}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Service = each.key
  }
}

# ── ECR Lifecycle Policy (keep only last 10 images per repo) ─────
resource "aws_ecr_lifecycle_policy" "service" {
  for_each   = aws_ecr_repository.service
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── IAM Role for EC2 (ECR pull + S3 read + CloudWatch) ───────────
resource "aws_iam_role" "ec2_role" {
  name = "deepfake-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ec2_ecr" {
  name = "deepfake-ec2-ecr-policy"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = [for repo in aws_ecr_repository.service : repo.arn]
      },
      {
        Sid    = "S3ModelRead"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}",
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "deepfake-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# ── Security Group: ALB (accepts HTTP from internet) ─────────────
resource "aws_security_group" "alb" {
  name        = "deepfake-alb-sg"
  description = "Allow HTTP/HTTPS inbound to ALB"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP from internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS from internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "deepfake-alb-sg" }
}

# ── Security Group: EC2 (accepts traffic from ALB + SSH) ─────────
resource "aws_security_group" "ec2" {
  name        = "deepfake-ec2-sg"
  description = "Allow traffic from ALB and SSH"
  vpc_id      = var.vpc_id

  # Frontend port — only from ALB
  ingress {
    description     = "Frontend from ALB"
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  # NOTE: API Gateway (port 5000) is NOT exposed via ALB.
  # The frontend container reaches api-gateway via Docker's deepfake-net network.

  # SSH from anywhere (restrict to your IP in production!)
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "deepfake-ec2-sg" }
}

# ── EC2 Instances ─────────────────────────────────────────────────
resource "aws_instance" "app" {
  count = 2

  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  subnet_id              = var.subnet_ids[count.index]
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  # Root volume: 30 GB is enough for Docker images
  root_block_device {
    volume_type           = "gp3"
    volume_size           = 30
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    aws_region       = var.aws_region
    aws_account_id   = var.aws_account_id
    ecr_registry     = local.ecr_registry
    github_owner     = var.github_owner
    github_repo      = var.github_repo
    s3_bucket_name   = var.s3_bucket_name
  })

  tags = {
    Name = "deepfake-app-${count.index + 1}"
    Role = "app-server"
  }
}

# ── Application Load Balancer ─────────────────────────────────────
resource "aws_lb" "main" {
  name               = "deepfake-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.subnet_ids

  enable_deletion_protection = false

  tags = { Name = "deepfake-alb" }
}

# ── Target Group: Frontend (:3000) ────────────────────────────────
resource "aws_lb_target_group" "frontend" {
  name        = "deepfake-frontend-tg"
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "instance"

  health_check {
    enabled             = true
    path                = "/"
    port                = "3000"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200-399"
  }

  tags = { Name = "deepfake-frontend-tg" }
}

# API Gateway target group REMOVED — api-gateway is internal only.
# Traffic flow: Internet → ALB → Frontend (port 3000)
#               Frontend container → api-gateway (Docker network, no ALB)
#               api-gateway → text-service / image-service (Docker network)

# ── Register EC2 instances in the frontend target group only ─────
resource "aws_lb_target_group_attachment" "frontend" {
  count            = 2
  target_group_arn = aws_lb_target_group.frontend.arn
  target_id        = aws_instance.app[count.index].id
  port             = 3000
}

# ── ALB Listener: HTTP :80 ────────────────────────────────────────
# Single rule: all traffic → frontend target group.
# The frontend container (Nginx) proxies /api/* to api-gateway internally.
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
}
# NOTE: The /api/* listener rule has been removed.
# Routing is now handled by Nginx inside the frontend container.
