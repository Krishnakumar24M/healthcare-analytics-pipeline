#!/usr/bin/env bash
# Deploy Streamlit Container to AWS ECS Fargate (ARM64 & AMD64 Compatible)
set -e

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="healthcare-risk-app"
CLUSTER_NAME="healthcare-cluster"
TASK_FAMILY="healthcare-risk-app"

echo "================================================================="
echo "🚀 AWS ECS Fargate Cloud Deployment"
echo "Target AWS Account: ${AWS_ACCOUNT_ID}"
echo "Target Region:     ${AWS_REGION}"
echo "Cluster:           ${CLUSTER_NAME}"
echo "================================================================="

# 1. Authenticate Docker to ECR
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 2. Tag & Push image
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
docker tag healthcare-risk-app:latest "${ECR_URI}"
docker push "${ECR_URI}"

# 3. Register Task Definition Revision
aws ecs register-task-definition --cli-input-json file://aws/ecs/task_definition.json --region "${AWS_REGION}" >/dev/null

# 4. Get default VPC subnets & security group
SUBNETS=$(aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" --query "Subnets[0:2].SubnetId" --output text | tr '\t' ',')
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text)

# Ensure port 8501 is open
aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" --protocol tcp --port 8501 --cidr 0.0.0.0/0 2>/dev/null || true

# 5. Launch Fargate Task
echo "Launching task on AWS ECS Fargate..."
TASK_ARN=$(aws ecs run-task \
  --cluster "${CLUSTER_NAME}" \
  --launch-type FARGATE \
  --task-definition "${TASK_FAMILY}" \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
  --region "${AWS_REGION}" \
  --query "tasks[0].taskArn" --output text)

echo "Task launched: ${TASK_ARN}"
echo "Waiting 30 seconds for task network interface..."
sleep 30

ENI_ID=$(aws ecs describe-tasks --cluster "${CLUSTER_NAME}" --tasks "${TASK_ARN}" --region "${AWS_REGION}" --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text)
PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "${ENI_ID}" --query "NetworkInterfaces[0].Association.PublicIp" --output text)

echo "================================================================="
echo "✅ Streamlit is LIVE on AWS Cloud!"
echo "🌐 URL: http://${PUBLIC_IP}:8501"
echo "================================================================="
