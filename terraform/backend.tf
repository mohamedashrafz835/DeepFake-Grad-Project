terraform {
  backend "s3" {
    bucket         = "statefile-bucket-2210xz"
    key            = "global/s3/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "statefile-table"
    encrypt        = true
  }
}
