# ☁️ Live AWS Cloud Infrastructure Deployment Guide

This guide bridges the gap between your verified local pipeline and production **AWS Live Cloud Infrastructure**.

---

## 🗺️ Local Environment vs. Live AWS Cloud Mapping

| Local Component | AWS Live Cloud Service | Production Benefit |
| :--- | :--- | :--- |
| **Docker Container (Port 8501)** | **AWS App Runner / ECS Fargate** | Fully managed containerized hosting with automatic TLS/HTTPS, autoscaling, and zero EC2 server management. |
| **Local CSV Data Files** | **Amazon S3 (`raw-data/`)** | Durable cloud object storage triggering event notifications (`s3:ObjectCreated:*`). |
| **Mock S3 Event Harness** | **AWS Lambda + EventBridge** | Event-driven serverless ETL executed at sub-second latency with zero idle server cost. |
| **Local Scikit-Learn / LightGBM** | **AWS SageMaker Endpoints & Pipelines** | Low-latency real-time HTTPS inference endpoints, model registry versioning, and auto-retraining. |

---

## 1. 🚀 AWS App Runner Container Deployment

AWS App Runner provides seamless, managed container hosting with an official public HTTPS URL and automatic SSL.

### Step 1.1: Push Docker Image to Amazon ECR
```bash
# Set your target AWS Region and Account ID
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/healthcare-risk-app"

# 1. Create ECR repository
aws ecr create-repository --repository-name healthcare-risk-app --region ${AWS_REGION} || true

# 2. Authenticate Docker with Amazon ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

# 3. Tag and push container image
docker tag healthcare-risk-app:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest
```

### Step 1.2: Launch App Runner Service
Use the automated deployment script provided in [`aws/deploy_to_apprunner.sh`](file:///Users/krishnakumarm/PROJECT/healthcare-analytics-pipeline/aws/deploy_to_apprunner.sh):
```bash
./aws/deploy_to_apprunner.sh
```

Or deploy via the AWS CLI using [`aws/apprunner_service.json`](file:///Users/krishnakumarm/PROJECT/healthcare-analytics-pipeline/aws/apprunner_service.json):
```bash
aws apprunner create-service --cli-input-json file://aws/apprunner_service.json --region us-east-1
```
> **Output**: App Runner generates a live public URL (e.g., `https://xyz123.us-east-1.awsapprunner.com`) hosting your Streamlit clinical application!

---

## 2. 🤖 AWS SageMaker Real-Time HTTPS Endpoint

Transitioning model inference from local `.joblib` files to an active AWS SageMaker HTTPS endpoint queried via `boto3`.

### Step 2.1: Package & Deploy Model Endpoint
Run the deployment script [`aws/sagemaker/deploy_sagemaker_endpoint.py`](file:///Users/krishnakumarm/PROJECT/healthcare-analytics-pipeline/aws/sagemaker/deploy_sagemaker_endpoint.py):
```bash
python3 aws/sagemaker/deploy_sagemaker_endpoint.py
```
This script:
1. Packages [`best_icu_model.joblib`](file:///Users/krishnakumarm/PROJECT/healthcare-analytics-pipeline/app/models/best_icu_model.joblib) into a `model.tar.gz` archive with an inference handler script.
2. Uploads the archive to `s3://[your-bucket]/sagemaker/models/`.
3. Creates a SageMaker Model, Endpoint Configuration, and deploys `icu-readmission-endpoint` on an `ml.t2.medium` instance.

### Step 2.2: Live Endpoint Querying in Streamlit
In your Streamlit application:
1. Open the left sidebar $\rightarrow$ expand **AWS SageMaker Live Endpoint**.
2. Check **Query Live SageMaker Endpoint**.
3. Enter your Endpoint Name (`icu-readmission-endpoint`) and Region (`us-east-1`).
4. Predictions will now query the live AWS SageMaker HTTPS endpoint via `boto3.client('sagemaker-runtime').invoke_endpoint()` with automatic fallback to local models if offline.

---

## 3. ⏱️ AWS EventBridge Automated Retraining Rules

To automate model retraining whenever raw clinical data accumulates in S3:

### Architecture Flow:
```text
[ New CSV Uploaded to S3: raw-data/ ]
                 │
                 ▼
     [ Amazon EventBridge Rule ]
                 │
                 ▼
 [ AWS Lambda: Check Record Volume ]
                 │
                 ▼ (If > 5,000 Records)
  [ AWS SageMaker CreateTrainingJob ] ──► [ Update Model in Registry ]
```

### Step 3.1: Enable EventBridge Notifications on S3 Bucket
```bash
aws s3api put-bucket-notification-configuration \
    --bucket healthcare-analytics-pipeline-[your-name] \
    --notification-configuration '{"EventBridgeConfiguration": {}}'
```

### Step 3.2: Create EventBridge Rule
Deploy the pattern defined in [`aws/eventbridge/eventbridge_rule_pattern.json`](file:///Users/krishnakumarm/PROJECT/healthcare-analytics-pipeline/aws/eventbridge/eventbridge_rule_pattern.json):
```bash
aws events put-rule \
    --name "TriggerHealthcareRetrainingOnUpload" \
    --event-pattern file://aws/eventbridge/eventbridge_rule_pattern.json \
    --state ENABLED
```

### Step 3.3: Deploy the Retraining Dispatcher Lambda
1. Create a Lambda function with the code from [`aws/eventbridge/trigger_retraining_lambda.py`](file:///Users/krishnakumarm/PROJECT/healthcare-analytics-pipeline/aws/eventbridge/trigger_retraining_lambda.py).
2. Set Environment Variables:
   - `RECORD_THRESHOLD`: `5000`
   - `SAGEMAKER_ROLE_ARN`: `arn:aws:iam::<ACCOUNT_ID>:role/SageMakerExecutionRole`
3. Add the EventBridge rule as an event source target to the Lambda function:
   ```bash
   aws events put-targets \
       --rule "TriggerHealthcareRetrainingOnUpload" \
       --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:<ACCOUNT_ID>:function:trigger-retraining-lambda"
   ```

---

## 📋 Summary Verification Checklist

- [x] **AWS App Runner**: Configured in `aws/apprunner_service.json` and `aws/deploy_to_apprunner.sh`.
- [x] **SageMaker Endpoints**: Python deployment script in `aws/sagemaker/deploy_sagemaker_endpoint.py` and live `boto3` client in `app/app.py`.
- [x] **EventBridge Retraining**: Event pattern rule in `aws/eventbridge/eventbridge_rule_pattern.json` and Lambda volume inspector in `aws/eventbridge/trigger_retraining_lambda.py`.
