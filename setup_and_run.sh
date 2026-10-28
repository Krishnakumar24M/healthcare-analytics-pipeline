#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🏥 Serverless Healthcare Analytics Pipeline Setup & Runner"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "1. Installing Python dependencies..."
python3 -m pip install -q -r requirements.txt

echo "2. Generating synthetic ICU Clinical Telemetry Data..."
python3 data/generate_sample_data.py

echo "3. Running Local Lambda Serverless ETL Cleansing..."
python3 lambda/test_lambda_local.py

echo "4. Training & Evaluating ML Models (Logistic Regression vs LightGBM)..."
python3 notebooks/train_and_evaluate.py

echo "5. Running Pytest CI Validation Suite..."
pytest tests/ -v

echo ""
echo "=========================================================="
echo "✅ Pipeline Build & Verification Completed Successfully!"
echo "=========================================================="
echo "To launch the Streamlit Web Application locally, run:"
echo "   streamlit run app/app.py"
echo ""
echo "To run with Docker:"
echo "   docker build -t healthcare-risk-app:latest ."
echo "   docker run -p 8501:8501 healthcare-risk-app:latest"
echo "=========================================================="
