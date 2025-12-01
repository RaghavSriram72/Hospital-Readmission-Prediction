# Predicting Hospital Readmission in Diabetes Patients

This project explores the use of kNN, Logistic Regression, Random Forest, and Neural Networks to predict hospital readmission in diabetes patients using clinical data and demographics.

## Overview

We evaluated four machine learning models to predict whether a patient will be readmitted to the hospital:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest**
- **Neural Network (MLP)**

on a **binary classification task**: Will the patient be readmitted to the hospital?

## Dataset

We used the **UCI Diabetes 130-US Hospitals dataset** from the UCI Machine Learning Repository.

The dataset contains:

- **101,766 patient records** from 130 US hospitals
- **50+ clinical features** including:
  - Demographic information (age, gender, race)
  - Clinical measurements (number of medications, lab procedures, diagnoses)
  - Hospital metrics (time in hospital, number of inpatient visits)
  - Medication information (metformin, insulin, etc.)
- **Target variable**: `readmitted` – whether the patient was readmitted within 30 days (YES/NO)

Data is fetched directly in the notebook using:

```python
from ucimlrepo import fetch_ucirepo
diabetes_130_us_hospitals = fetch_ucirepo(id=296)
```
## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
ucimlrepo
imbalanced-learn
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo imbalanced-learn
```

## Authors

- Raghav Sriram
- Yashas Raman
