variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "github_owner" {
  description = "GitHub username or organization that owns the repository"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name (without owner prefix)"
  type        = string
}

# ── EKS ──────────────────────────────────────────────────────────

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "deepfake-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.31"
}

variable "node_groups" {
  description = "Map of EKS managed node group configurations"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    scaling_config = object({
      desired_size = number
      max_size     = number
      min_size     = number
    })
  }))
  default = {
    "default" = {
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      scaling_config = {
        desired_size = 2
        max_size     = 3
        min_size     = 2
      }
    }
  }
}
