aws_region     = "us-east-1"
github_owner   = "mohamedashrafz835"
github_repo    = "DeepFake-Grad-Project"

cluster_name    = "deepfake-cluster"
cluster_version = "1.31"

node_groups = {
  "default" = {
    instance_types = ["m7i-flex.large"]
    capacity_type  = "ON_DEMAND"
    scaling_config = {
      desired_size = 2
      max_size     = 3
      min_size     = 2
    }
  }
}
