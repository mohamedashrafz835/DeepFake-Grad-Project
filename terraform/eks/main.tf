# ─────────────────────────────────────────────────────────────────
# EKS Module — Cluster Control Plane
#
# VPC config includes BOTH public + private subnets so the control
# plane can communicate with nodes and internet-facing LBs.
# ─────────────────────────────────────────────────────────────────

resource "aws_eks_cluster" "main" {
  name    = var.cluster_name
  version = var.cluster_version

  access_config {
    authentication_mode                         = "API_AND_CONFIG_MAP"
    bootstrap_cluster_creator_admin_permissions = true
  }

  role_arn = aws_iam_role.cluster.arn

  vpc_config {
    # Register both subnet tiers so the control plane endpoint is reachable
    # from both the ALB (public) and nodes (private)
    subnet_ids = concat(var.public_subnet_ids, var.private_subnet_ids)
  }

  tags = {
    Name = var.cluster_name
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
  ]
}

# ── IAM Role for EKS Control Plane ───────────────────────────────
resource "aws_iam_role" "cluster" {
  name = "${var.cluster_name}-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster.name
}