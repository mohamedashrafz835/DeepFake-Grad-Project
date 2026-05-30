# ─────────────────────────────────────────────────────────────────
# VPC Module
#
# Subnet layout (2 AZs):
#   Public  subnet A  10.0.1.0/24  →  ALB / internet-facing LB
#   Public  subnet B  10.0.2.0/24  →  ALB / internet-facing LB
#   Private subnet A  10.0.3.0/24  →  EKS worker nodes
#   Private subnet B  10.0.4.0/24  →  EKS worker nodes
#
# NAT Gateway (single, in public subnet A) gives worker nodes
# outbound internet access for ECR image pulls, SSM, etc.
# ─────────────────────────────────────────────────────────────────

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "deepfake-vpc"
  }
}

# ── Public Subnets ────────────────────────────────────────────────
resource "aws_subnet" "public" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                                        = "deepfake-public-subnet-${count.index + 1}"
    # Required by AWS Load Balancer Controller for internet-facing ALBs
    "kubernetes.io/role/elb"                    = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# ── Private Subnets ───────────────────────────────────────────────
resource "aws_subnet" "private" {
  count = length(var.private_subnet_cidrs)

  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.azs[count.index]

  tags = {
    Name                                        = "deepfake-private-subnet-${count.index + 1}"
    # Required by AWS Load Balancer Controller for internal ALBs
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# ── Internet Gateway (public subnets → internet) ──────────────────
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "deepfake-igw"
  }
}

# ── Elastic IP for NAT Gateway ────────────────────────────────────
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name = "deepfake-nat-eip"
  }

  depends_on = [aws_internet_gateway.igw]
}

# ── NAT Gateway (private subnets → internet, via public subnet A) ─
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id   # sits in public subnet A

  tags = {
    Name = "deepfake-nat-gw"
  }

  depends_on = [aws_internet_gateway.igw]
}

# ── Public Route Table ────────────────────────────────────────────
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "deepfake-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── Private Route Table ───────────────────────────────────────────
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name = "deepfake-private-rt"
  }
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}
