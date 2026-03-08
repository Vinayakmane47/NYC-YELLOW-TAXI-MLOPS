# AWS Deployment Guide (IaC with Terraform)

This document describes how to deploy the NYC Yellow Taxi MLOps system to AWS using Infrastructure as Code (Terraform).

## Target Architecture

```
                        +-----------------+
                        |   Route 53      |
                        |   (DNS)         |
                        +--------+--------+
                                 |
                        +--------v--------+
                        |   ALB           |
                        |   (HTTPS)       |
                        +--------+--------+
                                 |
                 +---------------+---------------+
                 |               |               |
          +------v------+ +-----v-------+ +-----v-------+
          | ECS Fargate | | ECS Fargate | | ECS Fargate |
          | inference   | | grafana     | | locust      |
          | :8000       | | :3000       | | :8089       |
          +------+------+ +-----+-------+ +-------------+
                 |               |
          +------v------+ +-----v-------+
          | Amazon      | | Amazon      |
          | Managed     | | Managed     |
          | Prometheus  | | Grafana     |
          +-------------+ +-------------+
                 |
          +------v------+
          | ECR         |      +------------------+
          | (images)    |      | Secrets Manager  |
          +-------------+      | (DAGSHUB_TOKEN)  |
                               +------------------+
```

## AWS Service Mapping

| Local Service | AWS Service | Why |
|--------------|-------------|-----|
| inference-api container | **ECS Fargate** | Serverless containers, no EC2 management |
| Docker images | **ECR** | Private container registry |
| Prometheus | **Amazon Managed Prometheus (AMP)** | Fully managed, scales automatically |
| Grafana | **Amazon Managed Grafana (AMG)** | SSO integration, managed upgrades |
| `.env` / DAGSHUB_TOKEN | **Secrets Manager** | Encrypted, auditable, rotatable |
| Load balancing | **ALB** | Layer 7, path-based routing, HTTPS |
| Networking | **VPC** | Isolated network with public/private subnets |
| Locust | **ECS Fargate (on-demand)** | Scale to 0 when not testing |
| MinIO | **S3** | Native object storage |
| PostgreSQL (Airflow) | **RDS PostgreSQL** | Managed database |
| Airflow | **Amazon MWAA** | Managed Airflow service |

## Terraform Project Structure

```
infra/
|-- main.tf                 # Provider config, Terraform backend
|-- variables.tf            # Input variables (region, env, etc.)
|-- outputs.tf              # ALB URL, Grafana URL, etc.
|-- backend.tf              # S3 + DynamoDB state locking
|
|-- modules/
|   |-- networking/
|   |   |-- main.tf         # VPC, subnets, NAT gateway, security groups
|   |   |-- variables.tf
|   |   `-- outputs.tf
|   |
|   |-- ecr/
|   |   |-- main.tf         # Container registries
|   |   `-- outputs.tf
|   |
|   |-- ecs/
|   |   |-- main.tf         # ECS cluster, task defs, services
|   |   |-- iam.tf          # Task execution roles
|   |   |-- variables.tf
|   |   `-- outputs.tf
|   |
|   |-- alb/
|   |   |-- main.tf         # ALB, target groups, listeners
|   |   |-- variables.tf
|   |   `-- outputs.tf
|   |
|   |-- monitoring/
|   |   |-- main.tf         # AMP workspace, AMG workspace
|   |   `-- outputs.tf
|   |
|   |-- secrets/
|   |   |-- main.tf         # Secrets Manager for DAGSHUB_TOKEN
|   |   `-- outputs.tf
|   |
|   |-- storage/
|   |   |-- main.tf         # S3 buckets (bronze, silver, gold, ml-transformed)
|   |   `-- outputs.tf
|   |
|   `-- airflow/
|       |-- main.tf         # MWAA environment
|       |-- variables.tf
|       `-- outputs.tf
|
`-- environments/
    |-- dev.tfvars           # Dev settings (smaller instances)
    `-- prod.tfvars          # Production settings
```

## Component Details

### 1. Networking (VPC)

```hcl
# 2 AZs for high availability
# Public subnets: ALB, NAT Gateway
# Private subnets: ECS tasks, RDS, MWAA

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
}

# Public subnets (2 AZs)
# Private subnets (2 AZs)
# NAT Gateway (for private subnet internet access)
# Security groups per service
```

**Security groups:**
- ALB: Inbound 80/443 from anywhere
- Inference API: Inbound 8000 from ALB only
- Prometheus: Inbound 9090 from Grafana and inference-api only
- RDS: Inbound 5432 from MWAA only

### 2. ECR (Container Registry)

```hcl
# One repository per image
resource "aws_ecr_repository" "inference" {
  name                 = "nyc-taxi-inference"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle_policy = "keep last 5 images"
}
```

Repositories needed:
- `nyc-taxi-inference` - Inference API image
- `nyc-taxi-airflow` - Custom Airflow image (if not using MWAA)

### 3. ECS Fargate (Inference API)

```hcl
resource "aws_ecs_service" "inference" {
  name            = "inference-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.inference.arn
  desired_count   = 2  # Multiple replicas for HA
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = module.networking.private_subnet_ids
    security_groups  = [module.networking.inference_sg_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = module.alb.inference_tg_arn
    container_name   = "inference-api"
    container_port   = 8000
  }
}
```

**Task definition:**
- CPU: 512 (0.5 vCPU) - RandomForest inference is lightweight
- Memory: 1024 MB (1 GB) - model + pandas overhead
- Health check: `GET /health`
- Secrets: `DAGSHUB_TOKEN` from Secrets Manager

**Auto-scaling:**
```hcl
resource "aws_appautoscaling_target" "inference" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "service/${cluster}/${service}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.inference.resource_id

  target_tracking_scaling_policy_configuration {
    target_value = 70.0
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
```

### 4. ALB (Load Balancer)

```hcl
resource "aws_lb" "main" {
  name               = "nyc-taxi-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [module.networking.alb_sg_id]
  subnets            = module.networking.public_subnet_ids
}

# HTTPS listener with ACM certificate
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  certificate_arn   = aws_acm_certificate.main.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.inference.arn
  }
}
```

### 5. Secrets Manager

```hcl
resource "aws_secretsmanager_secret" "dagshub_token" {
  name = "nyc-taxi/dagshub-token"
}

# Reference in ECS task definition:
# secrets = [{
#   name      = "DAGSHUB_TOKEN"
#   valueFrom = aws_secretsmanager_secret.dagshub_token.arn
# }]
```

### 6. Amazon Managed Prometheus (AMP)

```hcl
resource "aws_prometheus_workspace" "main" {
  alias = "nyc-taxi-inference"
}

# Configure inference API to remote-write to AMP
# Or use a Prometheus agent sidecar in the ECS task
```

The inference API's `/metrics` endpoint works as-is. Deploy a lightweight Prometheus agent as a sidecar container in the ECS task definition that scrapes `/metrics` and remote-writes to AMP.

### 7. Amazon Managed Grafana (AMG)

```hcl
resource "aws_grafana_workspace" "main" {
  name                     = "nyc-taxi-monitoring"
  account_access_type      = "CURRENT_ACCOUNT"
  authentication_providers = ["AWS_SSO"]
  permission_type          = "SERVICE_MANAGED"
  role_arn                 = aws_iam_role.grafana.arn

  data_sources = ["PROMETHEUS"]
}
```

Import the existing `monitoring/grafana/dashboards/inference.json` via the Grafana API or Terraform's `grafana_dashboard` resource.

### 8. S3 Buckets (Data Lake)

```hcl
locals {
  buckets = ["bronze", "silver", "gold", "ml-transformed"]
}

resource "aws_s3_bucket" "data" {
  for_each = toset(local.buckets)
  bucket   = "nyc-taxi-${each.key}-${var.environment}"
}

resource "aws_s3_bucket_versioning" "data" {
  for_each = aws_s3_bucket.data
  bucket   = each.value.id
  versioning_configuration { status = "Enabled" }
}
```

Update `src/config/settings.yaml` to use `s3://` paths instead of `s3a://` MinIO paths.

### 9. Amazon MWAA (Managed Airflow)

```hcl
resource "aws_mwaa_environment" "main" {
  name               = "nyc-taxi-airflow"
  airflow_version    = "2.8.2"
  environment_class  = "mw1.small"
  max_workers        = 2

  source_bucket_arn          = aws_s3_bucket.airflow_dags.arn
  dag_s3_path                = "dags/"
  requirements_s3_path       = "requirements.txt"

  network_configuration {
    security_group_ids = [module.networking.mwaa_sg_id]
    subnet_ids         = module.networking.private_subnet_ids
  }
}
```

Upload DAGs to S3 instead of mounting from local filesystem.

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push inference image
        run: |
          docker build -f Dockerfile.inference -t $ECR_REGISTRY/nyc-taxi-inference:${{ github.sha }} .
          docker push $ECR_REGISTRY/nyc-taxi-inference:${{ github.sha }}

      - name: Deploy to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: task-definition.json
          service: inference-api
          cluster: nyc-taxi-cluster
          wait-for-service-stability: true
```

## Cost Optimization

| Service | Cost Strategy |
|---------|--------------|
| ECS Fargate (inference) | Start with 0.5 vCPU / 1GB. Auto-scale 1-10 tasks based on CPU |
| ECS Fargate (Locust) | Use **Fargate Spot** (70% cheaper). Run only during testing |
| AMP | Free tier covers 200M samples ingested. Pay per query beyond that |
| AMG | ~$9/month per active editor |
| S3 | Use Intelligent-Tiering for bronze/silver (infrequent access) |
| MWAA | `mw1.small` (~$0.49/hr). Scale down to 1 worker in dev |
| NAT Gateway | Single NAT in one AZ for dev. Multi-AZ for prod |
| ALB | ~$0.023/hr + data transfer |

**Estimated monthly cost (dev):** ~$150-200/month
**Estimated monthly cost (prod):** ~$400-600/month

## Migration Checklist

1. **Terraform init:**
   ```bash
   cd infra/
   terraform init
   terraform plan -var-file=environments/dev.tfvars
   terraform apply -var-file=environments/dev.tfvars
   ```

2. **Push images to ECR:**
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
   docker build -f Dockerfile.inference -t $ECR_URI/nyc-taxi-inference:latest .
   docker push $ECR_URI/nyc-taxi-inference:latest
   ```

3. **Store secrets:**
   ```bash
   aws secretsmanager put-secret-value \
     --secret-id nyc-taxi/dagshub-token \
     --secret-string $DAGSHUB_TOKEN
   ```

4. **Update config paths:**
   - Change `s3a://` paths to `s3://` in `settings.yaml`
   - Remove MinIO-specific config
   - Update endpoint URLs for AWS services

5. **Upload DAGs to S3:**
   ```bash
   aws s3 sync airflow/dags/ s3://nyc-taxi-airflow-dags/dags/
   ```

6. **Import Grafana dashboard:**
   - Upload `monitoring/grafana/dashboards/inference.json` via AMG API
   - Configure AMP as datasource

7. **Verify:**
   ```bash
   curl https://your-domain.com/health
   curl -X POST https://your-domain.com/predict -H 'Content-Type: application/json' \
     -d '{"pickup_datetime":"2025-06-15T14:30:00","PULocationID":161,"DOLocationID":237,"trip_distance":3.5}'
   ```

## What Stays the Same

- All application code (`src/`) works without changes
- `Dockerfile.inference` builds the same image for ECR
- Grafana dashboard JSON imports directly into AMG
- Prometheus metrics scraping works with AMP
- Airflow DAGs run on MWAA without modification (just upload to S3)
- The only config changes are storage paths (`s3a://` -> `s3://`) and removing MinIO settings
