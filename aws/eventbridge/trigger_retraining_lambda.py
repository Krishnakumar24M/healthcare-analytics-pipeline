"""
AWS Lambda Function: Trigger SageMaker Automated Retraining
Invoked by AWS EventBridge when raw telemetry data volume crosses a threshold.
"""
import json
import logging
import os
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# Configuration
RECORD_THRESHOLD = int(os.getenv("RECORD_THRESHOLD", "5000"))
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::<AWS_ACCOUNT_ID>:role/SageMakerExecutionRole")
TRAINING_IMAGE = os.getenv("TRAINING_IMAGE_URI", "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3")

def get_raw_data_record_count(bucket, prefix="raw-data/"):
    """Estimates total raw patient records in S3 bucket prefix"""
    total_records = 0
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.csv'):
                # Read header/sample line count
                response = s3.head_object(Bucket=bucket, Key=obj['Key'])
                file_size_bytes = response['ContentLength']
                # Average record is ~40 bytes
                total_records += max(1, file_size_bytes // 40)
    return total_records

def lambda_handler(event, context):
    logger.info(f"EventBridge Retraining Event: {json.dumps(event)}")
    
    # Extract S3 bucket from event
    detail = event.get('detail', {})
    bucket = detail.get('bucket', {}).get('name', 'healthcare-analytics-pipeline')
    
    current_records = get_raw_data_record_count(bucket)
    logger.info(f"Current Raw S3 Records: {current_records} | Retraining Threshold: {RECORD_THRESHOLD}")
    
    if current_records < RECORD_THRESHOLD:
        msg = f"Record count ({current_records}) below threshold ({RECORD_THRESHOLD}). Skipping retraining."
        logger.info(msg)
        return {'statusCode': 200, 'body': msg}
        
    training_job_name = f"icu-readmission-retrain-{context.aws_request_id[:8]}"
    logger.info(f"Launching SageMaker Training Job: {training_job_name}")
    
    try:
        response = sagemaker.create_training_job(
            TrainingJobName=training_job_name,
            AlgorithmSpecification={
                'TrainingImage': TRAINING_IMAGE,
                'TrainingInputMode': 'File'
            },
            RoleArn=SAGEMAKER_ROLE,
            InputDataConfig=[
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f"s3://{bucket}/cleaned-data/",
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': f"s3://{bucket}/sagemaker/retrained-models/"
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 10
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 3600
            }
        )
        logger.info(f"SageMaker Training Job started: {response['TrainingJobArn']}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Automated SageMaker Retraining Triggered',
                'training_job_arn': response['TrainingJobArn'],
                'current_records': current_records
            })
        }
    except Exception as e:
        logger.error(f"Error launching SageMaker retraining: {str(e)}", exc_info=True)
        return {'statusCode': 500, 'error': str(e)}
