#!/usr/bin/env bash
# Deploy Healthcare Streamlit Container to AWS App Runner
set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "<AWS_ACCOUNT_ID>")
ECR_REPO_NAME="healthcare-risk-app"
SERVICE_NAME="healthcare-analytics-dashboard"

echo "================================================================="
echo "🚀 AWS App Runner Container Deployment"
echo "Target AWS Account: ${AWS_ACCOUNT_ID}"
echo "Target Region:     ${AWS_REGION}"
echo "ECR Repository:    ${ECR_REPO_NAME}"
echo "================================================================="

if [ "$AWS_ACCOUNT_ID" = "<AWS_ACCOUNT_ID>" ]; then
    echo "⚠️  AWS CLI credentials not detected or AWS_ACCOUNT_ID not set."
    echo "Please configure AWS CLI using: aws configure"
    exit 1
fi

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "1. Checking/Creating Amazon ECR Repository..."
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}"

echo "2. Authenticating Docker to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_URI}"

echo "3. Tagging and Pushing Docker Image..."
docker tag healthcare-risk-app:latest "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"

echo "4. Deploying to AWS App Runner..."
# Check if service already exists
SERVICE_ARN=$(aws apprunner list-services --region "${AWS_REGION}" --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text)

if [ -n "$SERVICE_ARN" ]; then
    echo "Updating existing App Runner service: ${SERVICE_ARN}"
    aws apprunner start-deployment --service-arn "${SERVICE_ARN}" --region "${AWS_REGION}"
else
    echo "Creating new App Runner service..."
    sed -i.bak "s/<AWS_ACCOUNT_ID>/${AWS_ACCOUNT_ID}/g; s/<AWS_REGION>/${AWS_REGION}/g" aws/apprunner_service.json && rm -f aws/apprunner_service.json.bak
    aws apprunner create-service --cli-input-json file://aws/apprunner_service.json --region "${AWS_REGION}"
fi

echo "================================================================="
echo "✅ App Runner deployment initiated successfully!"
echo "Check service status and retrieve your public HTTPS URL with:"
echo "   aws apprunner describe-service --service-arn <SERVICE_ARN> --query 'Service.ServiceUrl'"
echo "================================================================="
