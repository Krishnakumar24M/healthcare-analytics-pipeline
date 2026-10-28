"""
Realistic ICU Clinical Telemetry Data Generator
Mimics MIMIC-III ICU readmission and telemetry data patterns.
"""
import os
import numpy as np
import pandas as pd

def generate_icu_telemetry_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    patient_ids = [f"PT-{10000 + i}" for i in range(n_samples)]
    
    # Age: Bimodal distribution with higher probability for older adults in ICU
    age = np.clip(
        np.concatenate([
            np.random.normal(loc=68, scale=12, size=int(n_samples * 0.75)),
            np.random.normal(loc=38, scale=10, size=int(n_samples * 0.25))
        ]),
        18, 98
    ).astype(int)
    np.random.shuffle(age)
    
    # Heart rate (BPM): typical mean 82, std 18
    heart_rate = np.clip(np.random.normal(loc=84, scale=19, size=n_samples), 45, 175).astype(int)
    
    # Systolic blood pressure (mmHg): mean 128, std 24
    blood_pressure = np.clip(np.random.normal(loc=130, scale=22, size=n_samples), 80, 210).astype(int)
    
    # Blood glucose (mg/dL): mean 125, std 45
    glucose = np.clip(np.random.normal(loc=135, scale=48, size=n_samples), 65, 360).astype(int)
    
    # Underlying conditions (0 to 4: e.g. hypertension, diabetes, CHF, COPD)
    underlying_conditions = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.20, 0.35, 0.25, 0.15, 0.05])
    
    # ICU length of stay in days
    icu_stay_days = np.clip(np.random.exponential(scale=3.5, size=n_samples) + 1, 1, 30).round(1)
    
    # Readmission 30-day target probability calculation based on clinical risk factors
    # Higher age, abnormal vitals, and comorbidities increase readmission odds
    z = (
        0.035 * (age - 60) +
        0.025 * (heart_rate - 80) +
        0.018 * (blood_pressure - 120) +
        0.012 * (glucose - 110) +
        0.55 * underlying_conditions +
        0.08 * icu_stay_days -
        1.8
    )
    prob_readmitted = 1 / (1 + np.exp(-z))
    readmitted_30days = (np.random.uniform(0, 1, n_samples) < prob_readmitted).astype(int)
    
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': age,
        'heart_rate': heart_rate,
        'blood_pressure': blood_pressure,
        'glucose': glucose,
        'underlying_conditions': underlying_conditions,
        'icu_stay_days': icu_stay_days,
        'readmitted_30days': readmitted_30days
    })
    
    # Inject ~2% duplicates to validate the Lambda deduplication step
    duplicate_rows = df.sample(n=int(n_samples * 0.02), random_state=random_state)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Inject missing values (NaNs) in ~3% of numerical columns to test median imputation
    for col in ['heart_rate', 'blood_pressure', 'glucose', 'age']:
        mask = np.random.rand(len(df)) < 0.03
        df.loc[mask, col] = np.nan
        
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "patient_telemetry.csv")
    
    print(f"Generating realistic ICU telemetry dataset...")
    df = generate_icu_telemetry_data(n_samples=2000)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} records at: {output_path}")
    print("Summary of data:")
    print(df.info())
    print(df.head())
