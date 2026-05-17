provider "aws" {
  region = "us-east-1"
}

# -------------------------------
# Remote Backend Setup
# -------------------------------
resource "aws_s3_bucket" "state-bucket" {
  bucket = "statefile-bucket-2210xz"
  tags = {
    Name = "state-buckethawarey"
  }
}

resource "aws_dynamodb_table" "statefile-table" {
  name         = "statefile-table"
  hash_key     = "LockID"
  billing_mode = "PAY_PER_REQUEST"

  attribute {
    name = "LockID"
    type = "S"
  }
}
