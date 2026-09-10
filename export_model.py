import os
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# Constants matching Part A-D notebook
RANDOM_STATE = 42
TEST_SIZE = 0.20
DATASET_PATH = Path("200000_employee_dataset.csv")
MODELS_DIR = Path("models")

def clean_column_names(df):
    """Standardize column names to snake_case."""
    cleaned_cols = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_')
    )
    df.columns = cleaned_cols
    return df

def train_and_export():
    print(f"Loading dataset from {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH.resolve()}")
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Raw shape: {df.shape}")
    
    df = clean_column_names(df)
    target_col = "annual_salary_usd"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col]
    
    # Feature columns matching Part A-D selection
    excluded_cols = [target_col, 'employee_id']
    features = [c for c in df.columns if c not in excluded_cols]
    X = df[features]
    print(f"Features: {features}")
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Extract supported categories
    education_levels = sorted(df['education'].dropna().unique().tolist())
    
    # Preprocessing pipelines matching notebook exactly
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ], remainder='drop'
    )
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Model pipeline
    model = LinearRegression()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    print("Performing 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        pipeline, X_train, y_train, cv=kf,
        scoring=('neg_mean_absolute_error', 'r2'),
        return_train_score=False, n_jobs=-1
    )
    
    cv_mae_mean = float(-cv_scores['test_neg_mean_absolute_error'].mean())
    cv_mae_std = float(cv_scores['test_neg_mean_absolute_error'].std())
    cv_r2_mean = float(cv_scores['test_r2'].mean())
    cv_r2_std = float(cv_scores['test_r2'].std())
    
    print(f"CV R² Mean: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
    print(f"CV MAE Mean (USD): ${cv_mae_mean:.2f} (+/- ${cv_mae_std:.2f})")
    
    print("Fitting model on full training set...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    test_r2 = float(r2_score(y_test, y_pred))
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_mse = float(mean_squared_error(y_test, y_pred))
    test_rmse = float(np.sqrt(test_mse))
    
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE (USD): ${test_mae:.2f}")
    print(f"Test MSE (USD²): {test_mse:.2f}")
    print(f"Test RMSE (USD): ${test_rmse:.2f}")
    
    # Calculate empirical median projects per experience year for realistic inference mapping
    projects_by_exp = df.groupby('working_experience_years')['number_of_projects'].median().to_dict()
    projects_by_exp = {int(k): float(v) for k, v in projects_by_exp.items()}
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 1. Save pipeline
    pipeline_path = MODELS_DIR / "salary_prediction_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline safely saved to {pipeline_path.resolve()}")
    
    # 2. Save factual metadata
    standard_job_titles = [
        "Software Engineer",
        "Senior Software Engineer",
        "Data Scientist",
        "Data Analyst",
        "Machine Learning Engineer",
        "Product Manager",
        "DevOps Engineer",
        "QA Engineer",
        "UI/UX Designer",
        "Business Analyst",
        "Cloud Architect",
        "Engineering Manager"
    ]
    
    metadata = {
        "project_name": "SalarySense AI",
        "target_column": "Annual_Salary_USD",
        "currency": "USD",
        "feature_columns": [
            "Age",
            "Gender",
            "Education_Level",
            "Years_of_Experience",
            "Job_Title"
        ],
        "model_feature_columns": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "education_levels": education_levels,
        "job_titles": standard_job_titles,
        "gender_options": ["Male", "Female", "Other", "Prefer not to say"],
        "model_name": "Linear Regression",
        "test_r2": round(test_r2, 4),
        "test_mae_usd": round(test_mae, 2),
        "test_mse_usd": round(test_mse, 2),
        "test_rmse_usd": round(test_rmse, 2),
        "cv_r2_mean": round(cv_r2_mean, 4),
        "cv_mae_mean_usd": round(cv_mae_mean, 2),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "200000_employee_dataset.csv",
        "dataset_target": "Annual_Salary_USD",
        "default_usd_to_inr_rate": 83.00,
        "usd_to_inr_rate_source": "Manually configured reference rate",
        "projects_by_exp": projects_by_exp
    }
    
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path.resolve()}")
    print("Model export completed successfully!")

if __name__ == "__main__":
    train_and_export()
