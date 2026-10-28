"""
AWS Lambda Serverless ETL Function
Triggered by S3 'ObjectCreated' events on 'raw-data/*.csv'
Cleans, normalizes, imputes missing values, and writes processed data to 'cleaned-data/'.
"""
import io
import json
import logging
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    import boto3
    s3_client = boto3.client('s3')
except ImportError:
    boto3 = None
    s3_client = None

def process_telemetry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core data cleaning & feature normalization pipeline.
    Pure function to facilitate unit testing without AWS dependencies.
    """
    initial_rows = len(df)

    
    # 1. Deduplication
    df = df.drop_duplicates().copy()
    logger.info(f"Deduplication removed {initial_rows - len(df)} duplicate records.")
    
    # 2. Identify numerical columns for imputation & normalization
    standard_clinical_cols = [
        'heart_rate', 'blood_pressure', 'glucose', 'age',
        'last_glucose', 'last_hemoglobin', 'last_creatinine',
        'bmi', 'weight_kg', 'height_cm', 'medications_count',
        'num_previous_admissions', 'length_of_stay', 'procedures_count'
    ]
    numerical_cols = [c for c in standard_clinical_cols if c in df.columns]
    if not numerical_cols:
        numerical_cols = list(df.select_dtypes(include=['number']).columns)

    # 3. Impute missing clinical values using median
    for col in numerical_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # 4. Feature Scaling (Min-Max Normalization to [0, 1])
    for col in numerical_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f'{col}_norm'] = 0.0

    # 5. Target mapping if using categorical risk labels (e.g. Kaggle 'High', 'Medium', 'Low')
    if 'readmission_risk' in df.columns and 'readmitted_30days' not in df.columns:
        df['readmitted_30days'] = df['readmission_risk'].apply(
            lambda x: 1 if str(x).strip().lower() in ['high', 'medium', '1', 'yes'] else 0
        )

    return df


def lambda_handler(event, context):
    """
    AWS Lambda entry point for S3 event notifications.
    """
    logger.info(f"Received S3 event: {json.dumps(event)}")
    
    try:
        # Extract bucket and key from S3 event structure
        records = event.get('Records', [])
        if not records:
            return {
                'statusCode': 400,
                'body': json.dumps('No records found in S3 event.')
            }
            
        record = records[0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        logger.info(f"Target Bucket: {bucket}, Object Key: {key}")
        
        # Validation: only process CSV files in raw-data/
        if not key.startswith('raw-data/') or not key.endswith('.csv'):
            logger.warning(f"Skipping key '{key}': Does not match prefix 'raw-data/' or suffix '.csv'")
            return {
                'statusCode': 200,
                'body': json.dumps(f"Skipping non-raw CSV file: {key}")
            }
        
        # Load raw CSV from S3
        logger.info(f"Downloading raw object s3://{bucket}/{key}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        raw_content = response['Body'].read()
        df_raw = pd.read_csv(io.BytesIO(raw_content))
        logger.info(f"Successfully loaded CSV with {len(df_raw)} rows and {len(df_raw.columns)} columns.")
        
        # Process and clean the dataframe
        df_cleaned = process_telemetry_dataframe(df_raw)
        
        # Prepare output keys
        # We save both cleaned CSV and Parquet for QuickSight & SageMaker compatibility
        output_csv_key = key.replace('raw-data/', 'cleaned-data/').replace('.csv', '_clean.csv')
        
        # Export Cleaned CSV
        csv_buffer = io.StringIO()
        df_cleaned.to_csv(csv_buffer, index=False)
        s3_client.put_object(
            Bucket=bucket,
            Key=output_csv_key,
            Body=csv_buffer.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )
        logger.info(f"Successfully exported cleaned CSV to s3://{bucket}/{output_csv_key}")
        
        # Optional Parquet export if pyarrow is present in the Lambda layer
        try:
            parquet_buffer = io.BytesIO()
            df_cleaned.to_parquet(parquet_buffer, index=False, engine='auto')
            output_parquet_key = key.replace('raw-data/', 'cleaned-data/').replace('.csv', '_clean.parquet')
            s3_client.put_object(
                Bucket=bucket,
                Key=output_parquet_key,
                Body=parquet_buffer.getvalue(),
                ContentType='application/x-parquet'
            )
            logger.info(f"Successfully exported Parquet to s3://{bucket}/{output_parquet_key}")
        except Exception as pe:
            logger.warning(f"Parquet conversion skipped (optional pyarrow dependency): {str(pe)}")
            
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f"Successfully processed {key} to {output_csv_key}",
                'records_processed': len(df_cleaned),
                'columns': list(df_cleaned.columns)
            })
        }
        
    except Exception as e:
        logger.error(f"Error executing Lambda ETL handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
