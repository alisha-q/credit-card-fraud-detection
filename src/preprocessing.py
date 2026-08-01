"""
Reusable preprocessing functions for the Explainable Credit Card Fraud Detection project.

Mirrors the structure of the prior leukemia-detection-deep-learning project's src/preprocessing.py:
small, composable, well-documented functions that are imported directly into each notebook.

These functions implement Module 1 (Data Ingestion & Preprocessing) from the Interim Report's
High-Level Design: scaling of Time/Amount, stratified 70/15/15 splitting, and SMOTE oversampling
of the training set only (test/validation sets are left at the real-world class ratio).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def load_data(csv_path):
    """Load the creditcard.csv dataset and do basic sanity checks."""
    df = pd.read_csv(csv_path)
    assert TARGET_COLUMN in df.columns, "Expected a 'Class' column (0=legitimate, 1=fraud)"
    assert df[TARGET_COLUMN].isin([0, 1]).all(), "Class column must be binary"
    return df


def scale_time_amount(X_train, X_val, X_test, feature_columns=FEATURE_COLUMNS):
    """
    Fit a StandardScaler on Time & Amount using ONLY the training set, then apply it to
    train/val/test. V1-V28 are already PCA-scaled by the data provider and are left untouched.
    """
    scaler = StandardScaler()
    cols_to_scale = ["Time", "Amount"]
    idx = [feature_columns.index(c) for c in cols_to_scale]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[:, idx] = scaler.fit_transform(X_train[:, idx])
    X_val[:, idx] = scaler.transform(X_val[:, idx])
    X_test[:, idx] = scaler.transform(X_test[:, idx])

    return X_train, X_val, X_test, scaler


def stratified_split(df, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN,
                      train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Stratified 70/15/15 split that preserves the fraud/legitimate ratio in every split,
    matching Module 1 of the Interim Report's High-Level Design.
    """
    X = df[feature_columns].values
    y = df[target_column].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, sampling_strategy=0.5, k_neighbors=5, random_state=42):
    """
    Apply SMOTE to the TRAINING SET ONLY (validation/test sets must reflect real-world
    class imbalance, per the Interim Report's Module 1 design).

    sampling_strategy=0.5 means the minority (fraud) class is oversampled to 50% of the
    majority (legitimate) class count -- a common practical middle ground between full 1:1
    balancing (which can encourage overfitting on synthetic points) and no oversampling at all.
    Combine with class_weight='balanced' on the classifier for extra imbalance robustness.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def full_preprocessing_pipeline(csv_path, sampling_strategy=0.5, random_state=42):
    """
    Convenience wrapper that runs the entire Module 1 pipeline end to end:
    load -> stratified split -> scale Time/Amount -> SMOTE the training set.

    Returns a dict with train/val/test arrays plus the fitted scaler for reuse in the app.
    """
    df = load_data(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        df, random_state=random_state
    )
    X_train, X_val, X_test, scaler = scale_time_amount(X_train, X_val, X_test)
    X_train_resampled, y_train_resampled = apply_smote(
        X_train, y_train, sampling_strategy=sampling_strategy, random_state=random_state
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_train_resampled": X_train_resampled, "y_train_resampled": y_train_resampled,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
    }
