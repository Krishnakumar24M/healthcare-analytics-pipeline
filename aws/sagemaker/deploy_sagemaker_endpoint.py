"""
Deploy Trained ICU Readmission Model as an AWS SageMaker Real-Time HTTPS Endpoint.
Packages the model artifact, registers it in SageMaker Model Registry, and hosts an endpoint.
"""
import os
import tarfile
import json
import boto3

try:
    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel
    HAS_SAGEMAKER_SDK = True
except ImportError:
    HAS_SAGEMAKER_SDK = False

# Configuration
REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("S3_BUCKET", "healthcare-analytics-pipeline-306451251863")
ENDPOINT_NAME = "icu-readmission-endpoint"
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::306451251863:role/SageMakerExecutionRole")


def create_model_tar():
    """Package the model and inference entrypoint into model.tar.gz for SageMaker"""
    model_tar_path = "model.tar.gz"
    model_source = "app/models/best_icu_model.joblib"
    
    inference_script = """
import os
import joblib
import json
import pandas as pd
import numpy as np

def model_fn(model_dir):
    bundle = joblib.load(os.path.join(model_dir, 'best_icu_model.joblib'))
    return bundle

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame(data['instances'])
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_bundle):
    model = model_bundle['model']
    probs = model.predict_proba(input_data)[:, 1]
    return probs.tolist()

def output_fn(prediction, content_type):
    return json.dumps({'readmission_probabilities': prediction}), 'application/json'
"""
    with open("inference.py", "w") as f:
        f.write(inference_script.strip())

    with tarfile.open(model_tar_path, "w:gz") as tar:
        tar.add(model_source, arcname="best_icu_model.joblib")
        tar.add("inference.py", arcname="code/inference.py")
        
    os.remove("inference.py")
    print(f"Created {model_tar_path}")
    return model_tar_path

def deploy_endpoint():
    print(f"Deploying SageMaker Real-Time Endpoint: {ENDPOINT_NAME}")
    
    # Check credentials
    session = boto3.Session(region_name=REGION)
    credentials = session.get_credentials()
    if not credentials:
        print("⚠️  No AWS credentials found. Please run 'aws configure' or export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        return None

    model_tar = create_model_tar()
    s3_client = session.client('s3')
    sm_client = session.client('sagemaker')
    
    s3_key = f"sagemaker/models/{model_tar}"
    print(f"Uploading {model_tar} to s3://{BUCKET}/{s3_key}...")
    try:
        s3_client.upload_file(model_tar, BUCKET, s3_key)
        s3_model_uri = f"s3://{BUCKET}/{s3_key}"
        print(f"Uploaded model artifact to: {s3_model_uri}")
    except Exception as e:
        print(f"Error uploading to S3 ({e}). Ensure bucket '{BUCKET}' exists and credentials have s3:PutObject permissions.")
        return None

    if HAS_SAGEMAKER_SDK:
        sagemaker_session = sagemaker.Session(boto_session=session)
        role = ROLE_ARN if ROLE_ARN and "<AWS_ACCOUNT_ID>" not in ROLE_ARN else sagemaker.get_execution_role()
        sklearn_model = SKLearnModel(
            model_data=s3_model_uri,
            role=role,
            entry_point="code/inference.py",
            framework_version="1.2-1",
            py_version="py3",
            name="icu-readmission-lgbm"
        )
        predictor = sklearn_model.deploy(
            initial_instance_count=1,
            instance_type="ml.t2.medium",
            endpoint_name=ENDPOINT_NAME
        )
        print(f"✅ SageMaker Real-Time Endpoint Live: {predictor.endpoint_name}")
        return predictor
    else:
        print("Using standard boto3 client to configure SageMaker endpoint...")
        # Pure boto3 creation
        image_uri = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
        model_name = f"icu-readmission-model"
        
        try:
            sm_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': s3_model_uri,
                    'Environment': {'SAGEMAKER_PROGRAM': 'code/inference.py'}
                },
                ExecutionRoleArn=ROLE_ARN
            )
            config_name = f"{ENDPOINT_NAME}-config"
            sm_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.t2.medium',
                    'InitialVariantWeight': 1.0
                }]
            )
            resp = sm_client.create_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=config_name
            )
            print(f"✅ SageMaker Endpoint Creation Initiated: {resp['EndpointArn']}")
            return resp
        except Exception as e:
            print(f"Error creating SageMaker endpoint: {e}")
            return None
    return predictor

if __name__ == "__main__":
    deploy_endpoint()
