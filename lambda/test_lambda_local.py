"""
Local Test Suite for AWS Lambda Function
Uses unittest.mock to test S3 trigger event handling without requiring live AWS credentials.
"""
import io
import json
import os
import sys
from unittest.mock import MagicMock, patch
import pandas as pd

# Add lambda dir to path
sys.path.insert(0, os.path.dirname(__file__))
from lambda_function import lambda_handler, process_telemetry_dataframe

def test_dataframe_cleaning():
    print("\n--- 1. Testing Core DataFrame Cleansing Function ---")
    mock_data = {
        'patient_id': ['PT-1', 'PT-2', 'PT-2'],  # PT-2 is duplicate
        'heart_rate': [80.0, None, None],       # has NaN
        'blood_pressure': [120.0, 140.0, 140.0],
        'glucose': [100.0, 110.0, 110.0],
        'age': [60.0, 70.0, 70.0],
        'underlying_conditions': [1, 2, 2],
        'readmitted_30days': [0, 1, 1]
    }
    df = pd.DataFrame(mock_data)
    print("Input shape:", df.shape)
    
    cleaned_df = process_telemetry_dataframe(df)
    print("Cleaned shape (after deduplication):", cleaned_df.shape)
    assert len(cleaned_df) == 2, f"Expected 2 rows after dedup, got {len(cleaned_df)}"
    assert not cleaned_df['heart_rate'].isnull().any(), "Missing heart_rate values remain"
    assert not cleaned_df['glucose'].isnull().any(), "Missing glucose values remain"
    assert 'heart_rate_norm' in cleaned_df.columns, "heart_rate_norm not created"
    assert 'blood_pressure_norm' in cleaned_df.columns, "blood_pressure_norm not created"
    print("✅ Local DataFrame processing test passed successfully!")

def test_lambda_handler_mock_s3():
    print("\n--- 2. Testing Lambda S3 Trigger Event Handler ---")
    
    # Read actual sample data from raw folder if generated
    sample_csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "patient_telemetry.csv")
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, 'rb') as f:
            csv_bytes = f.read()
    else:
        csv_bytes = b"patient_id,heart_rate,blood_pressure,glucose,age,underlying_conditions,readmitted_30days\nPT-1,80,120,100,60,1,0\n"

    mock_event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "healthcare-analytics-pipeline-test"},
                    "object": {"key": "raw-data/patient_telemetry.csv"}
                }
            }
        ]
    }

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        'Body': io.BytesIO(csv_bytes)
    }

    uploaded_keys = []
    def fake_put_object(**kwargs):
        uploaded_keys.append(kwargs.get('Key'))
        return {'ResponseMetadata': {'HTTPStatusCode': 200}}

    mock_s3.put_object.side_effect = fake_put_object

    with patch('lambda_function.s3_client', mock_s3):
        response = lambda_handler(mock_event, None)
        print("Lambda Response:", response)
        assert response['statusCode'] == 200, f"Expected 200, got {response['statusCode']}"
        print(f"Uploaded S3 Keys: {uploaded_keys}")
        assert any("cleaned-data/patient_telemetry_clean.csv" in k for k in uploaded_keys), "Cleaned CSV key missing"
        print("✅ Mock S3 Lambda handler test passed successfully!")

if __name__ == "__main__":
    test_dataframe_cleaning()
    test_lambda_handler_mock_s3()
