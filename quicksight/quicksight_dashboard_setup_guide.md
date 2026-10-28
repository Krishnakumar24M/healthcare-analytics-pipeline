# 📊 AWS QuickSight Visualization Dashboard Setup Guide

This guide ensures full marks for the **Visualization** sub-rubric by connecting AWS QuickSight directly to your S3 cleaned dataset.

---

## 1. Prerequisites & S3 Permission Configuration
1. Open the **AWS QuickSight Console**.
2. If this is your first time, sign up for QuickSight (Standard or Enterprise edition).
3. Ensure QuickSight has permissions to read your S3 bucket:
   - Go to **Manage QuickSight** (top right profile menu) $\rightarrow$ **Security & permissions**.
   - Under **QuickSight access to AWS services**, click **Add or remove**.
   - Check the box for **Amazon S3** and select `healthcare-analytics-pipeline-[your-name]`.
   - Click **Update**.

---

## 2. Ingesting Processed Data into QuickSight
1. In QuickSight, navigate to **Datasets** $\rightarrow$ **New dataset**.
2. Select **S3** as the data source.
3. Configure the source:
   - **Data source name**: `Healthcare-Cleaned-Telemetry`
   - **Upload a manifest file**: Select **Upload** and choose `quicksight/quicksight_manifest.json` (remember to update `[your-name]` to your actual bucket name).
4. Click **Connect**, then choose **Direct query your data** or **Import to SPICE for quicker analytics**.
5. Click **Visualize**.

---

## 3. Creating Calculated Fields (Required for Rubric Visuals)

In the analysis view, click **Add** $\rightarrow$ **Add calculated field**:

### Field 1: Age Bracket
- **Field name**: `Age_Bracket`
- **Formula**:
  ```text
  ifelse({age} < 40, '<40 Years',
  ifelse({age} >= 40 AND {age} <= 60, '40-60 Years',
  '60+ Years'))
  ```

### Field 2: Readmission Status Label
- **Field name**: `Readmission_Status`
- **Formula**:
  ```text
  ifelse({readmitted_30days} = 1, 'Readmitted (High Risk)', 'Discharged (Normal)')
  ```

### Field 3: Readmission Percentage
- **Field name**: `Readmission_Rate_Pct`
- **Formula**:
  ```text
  avg({readmitted_30days}) * 100
  ```

---

## 4. Visual Components Checklist (Rubric Requirements)

### 📌 Visual 1: KPI Gauge (Overall High-Risk Patient Percentage)
1. Add a new visual and choose the **Gauge** visual type.
2. **Value**: Drag `readmitted_30days` into Value (set aggregation to **Average**).
3. **Format**: Format the field as a Percentage (e.g., `32.5%`).
4. **Target / Thresholds**: Set green $<25\%$, amber $25\%\text{–}40\%$, red $>40\%$.

### 📌 Visual 2: Bar Chart (Average Readmission Probability Grouped by Age Bracket)
1. Select **Clustered Bar Chart** or **Vertical Bar Chart**.
2. **X-axis**: `Age_Bracket` ($<40$, $40\text{–}60$, $60+$).
3. **Value**: `readmitted_30days` (Aggregate: **Average**).
4. **Sort**: Sort by Age Bracket ascending.
5. **Data labels**: Turn on data labels to show percentage per age group.

### 📌 Visual 3: Scatter Plot (Heart Rate vs. Systolic Blood Pressure)
1. Select the **Scatter Plot** visual type.
2. **X-axis**: `blood_pressure` (Systolic BP in mmHg).
3. **Y-axis**: `heart_rate` (BPM).
4. **Color / Group**: `Readmission_Status` (Red for Readmitted, Green for Normal).
5. **Size**: `underlying_conditions` (optional: larger dots indicate more comorbidities).

### 📌 Visual 4: Interactive Slicers / Filters
1. Click **Filter** on the left navigation panel $\rightarrow$ **Add filter**.
2. Choose `Age_Bracket`.
3. Set filter control type to **Dropdown - Multi-select**.
4. Check **Apply to all visuals** on this sheet.

---

## 5. Publishing Dashboard
1. Click **Share** (top right) $\rightarrow$ **Publish dashboard**.
2. Name it: `ICU Telemetry Readmission Risk Intelligence`.
3. Capture high-resolution screenshots for your final Project Report!
