terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket = "blockchain-ml-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "Blockchain-ML-Platform"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC Configuration
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"
  
  name = "blockchain-ml-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway     = true
  single_nat_gateway     = false
  one_nat_gateway_per_az = true
  
  enable_vpn_gateway = false
  
  tags = {
    Environment = var.environment
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  cluster_name    = "blockchain-ml-cluster"
  cluster_version = "1.27"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_public_access = true
  
  # Node Groups
  eks_managed_node_groups = {
    general = {
      desired_size = 3
      min_size     = 1
      max_size     = 10
      
      instance_types = ["m5.large", "m5a.large", "m5d.large"]
      capacity_type  = "ON_DEMAND"
      
      labels = {
        role = "general"
      }
      
      tags = {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/blockchain-ml-cluster" = "owned"
      }
    }
    
    gpu = {
      desired_size = 2
      min_size     = 0
      max_size     = 5
      
      instance_types = ["g4dn.xlarge", "p3.2xlarge"]
      capacity_type  = "SPOT"
      
      labels = {
        role = "gpu"
        "node.kubernetes.io/instance-type" = "gpu"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
      
      tags = {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/blockchain-ml-cluster" = "owned"
      }
    }
  }
  
  tags = {
    Environment = var.environment
  }
}

# RDS PostgreSQL
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"
  
  identifier = "blockchain-ml-db"
  
  engine               = "postgres"
  engine_version       = "15.3"
  family               = "postgres15"
  major_engine_version = "15"
  
  instance_class      = "db.t3.large"
  allocated_storage   = 100
  storage_encrypted   = true
  storage_type        = "gp3"
  iops                = 3000
  
  db_name  = "mlmodels"
  username = var.db_username
  password = var.db_password
  port     = 5432
  
  multi_az               = true
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [module.security_group_rds.security_group_id]
  
  backup_retention_period = 7
  skip_final_snapshot     = false
  deletion_protection     = true
  
  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  
  parameters = [
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    },
    {
      name  = "pg_stat_statements.track"
      value = "all"
    }
  ]
  
  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "blockchain-ml-redis"
  engine              = "redis"
  node_type           = "cache.t3.medium"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379
  security_group_ids  = [module.security_group_redis.security_group_id]
  subnet_group_name   = aws_elasticache_subnet_group.redis.name
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "blockchain-ml-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

# S3 Buckets
resource "aws_s3_bucket" "model_storage" {
  bucket = "blockchain-ml-model-storage-${var.environment}"
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id
  
  rule {
    id     = "archive-old-models"
    status = "Enabled"
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
}

# SageMaker Notebook
resource "aws_sagemaker_notebook_instance" "ml_notebook" {
  name          = "blockchain-ml-notebook"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = "ml.t3.medium"
  
  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_configuration.notebook_lifecycle.name
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "notebook_lifecycle" {
  name = "blockchain-ml-lifecycle-config"
  
  on_create {
    content = base64encode(<<EOF
#!/bin/bash
sudo -u ec2-user -i <<'EOF2'
pip install --upgrade pip
pip install -r /home/ec2-user/SageMaker/requirements.txt
EOF2
EOF
    )
  }
}

# Lambda Functions
resource "aws_lambda_function" "model_training" {
  filename      = "lambda_functions/model_training.zip"
  function_name = "blockchain-ml-training"
  role          = aws_iam_role.lambda_role.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  timeout       = 900
  memory_size   = 1024
  
  environment {
    variables = {
      DB_HOST     = module.rds.db_instance_address
      DB_NAME     = module.rds.db_instance_name
      S3_BUCKET   = aws_s3_bucket.model_storage.bucket
      ENVIRONMENT = var.environment
    }
  }
  
  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [module.security_group_lambda.security_group_id]
  }
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_lambda_function" "model_verification" {
  filename      = "lambda_functions/model_verification.zip"
  function_name = "blockchain-ml-verification"
  role          = aws_iam_role.lambda_role.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  timeout       = 300
  memory_size   = 512
  
  environment {
    variables = {
      ETHEREUM_RPC_URL = var.ethereum_rpc_url
      CONTRACT_ADDRESS = var.contract_address
    }
  }
  
  tags = {
    Environment = var.environment
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "api" {
  name          = "blockchain-ml-api"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["*"]
    allow_headers = ["*"]
  }
}

resource "aws_apigatewayv2_stage" "stage" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = var.environment
  auto_deploy = true
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id             = aws_apigatewayv2_api.api.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.model_training.invoke_arn
  integration_method = "POST"
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "distribution" {
  origin {
    domain_name = aws_apigatewayv2_stage.stage.invoke_url
    origin_id   = "api-gateway"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "Blockchain ML API Distribution"
  default_root_object = ""
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "api-gateway"
    
    forwarded_values {
      query_string = true
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
  
  tags = {
    Environment = var.environment
  }
}

# Security Groups
module "security_group_rds" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 4.0"
  
  name        = "rds-security-group"
  description = "Security group for RDS"
  vpc_id      = module.vpc.vpc_id
  
  ingress_with_cidr_blocks = [
    {
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      description = "PostgreSQL access from EKS"
      cidr_blocks = module.vpc.vpc_cidr_block
    }
  ]
  
  egress_with_cidr_blocks = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = "0.0.0.0/0"
    }
  ]
}

module "security_group_redis" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 4.0"
  
  name        = "redis-security-group"
  description = "Security group for Redis"
  vpc_id      = module.vpc.vpc_id
  
  ingress_with_cidr_blocks = [
    {
      from_port   = 6379
      to_port     = 6379
      protocol    = "tcp"
      description = "Redis access from EKS"
      cidr_blocks = module.vpc.vpc_cidr_block
    }
  ]
}

module "security_group_lambda" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 4.0"
  
  name        = "lambda-security-group"
  description = "Security group for Lambda functions"
  vpc_id      = module.vpc.vpc_id
  
  egress_with_cidr_blocks = [
    {
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = "0.0.0.0/0"
    }
  ]
}

# IAM Roles
resource "aws_iam_role" "sagemaker_role" {
  name = "blockchain-ml-sagemaker-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role" "lambda_role" {
  name = "blockchain-ml-lambda-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_vpc" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# Outputs
output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  value = module.rds.db_instance_address
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "s3_bucket_name" {
  value = aws_s3_bucket.model_storage.bucket
}

output "api_gateway_url" {
  value = aws_apigatewayv2_stage.stage.invoke_url
}

output "cloudfront_url" {
  value = aws_cloudfront_distribution.distribution.domain_name
}