import pandas as pd
import numpy as np

import boto3
from pyathena import connect
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup, FeatureDefinition, FeatureTypeEnum
from sagemaker.session import Session
from sagemaker import get_execution_role

from pyathena import connect

import time
import shap
import json
import joblib

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, precision_score, recall_score

from itertools import combinations
import pickle
import warnings


def get_categorical_columns_from_s3(bucket_name, s3_client):
    """
    Fetches categorical column names and their unique category counts from the S3 stored encoding JSON.
    
    Returns:
        - categorical_columns (list): List of categorical feature names.
        - category_counts (dict): Dictionary where keys are categorical columns and values are the number of unique categories.
    """
    encodings_key = "encodings/encodings.json"
    encodings_file = "encodings.json"

    # Download the encodings file from S3
    s3_client.download_file(bucket_name, encodings_key, encodings_file)

    # Load the encodings JSON
    with open(encodings_file, "r") as f:
        label_encoders = json.load(f)

    # The categorical columns are simply the keys in this JSON
    categorical_columns = list(label_encoders.keys())

    # Compute number of unique categories per categorical column
    category_counts = {col: len(label_encoders[col]) for col in categorical_columns}

    print(f"Identified categorical columns: {categorical_columns}")
    print(f"Category counts per categorical column: {category_counts}")

    return categorical_columns, category_counts

def eval_model(model, X_test, y_test):
    """
    Evaluates the XGBoost model and returns evaluation metrics.
    Ensures all relevant metrics are logged for the model card.
    """
    # Convert test data into DMatrix
    dmatrix_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    
    # Make predictions
    y_pred_proba = model.predict(dmatrix_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions
    
    # Compute evaluation metrics
    test_log_loss = log_loss(y_test, y_pred_proba)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"✅ Test Log Loss: {test_log_loss:.4f}")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ Test AUC: {test_auc:.4f}")
    print(f"✅ Test Precision: {test_precision:.4f}")
    print(f"✅ Test Recall: {test_recall:.4f}")

    return test_accuracy, test_auc, test_precision, test_recall

def convert_object_to_category(df):
    """
    Converts all object columns in a pandas DataFrame to category dtype.
    """
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].astype('category')
    return df

def encode_categorical(df, categorical_columns):
    """
    Encodes categorical columns using Label Encoding and saves mappings.

    Parameters:
        df (pd.DataFrame): DataFrame containing categorical columns.
        categorical_columns (list): List of categorical column names.

    Returns:
        df (pd.DataFrame): DataFrame with encoded categorical columns.
        encoders (dict): Dictionary containing LabelEncoders for each column.
    """
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding
        encoders[col] = {category: int(code) for category, code in zip(le.classes_, le.transform(le.classes_))}
    return df, encoders

def shap_feature_engineering(model, X_train, X_test, bucket_name, s3_client, top_k=17, shap_threshold=0.01):
    """
    Uses SHAP to find important features and applies interaction rules.
    - Numeric × Numeric → Standard multiplication.
    - Categorical × Categorical → Concatenation.
    - 🚫 **DO NOT COMBINE categorical features with numeric features.**
    - Saves transformation rules and categorical encodings for future API inference.
    """
    # ✅ Step 1: Get categorical columns & category counts from S3
    categorical_columns, category_counts = get_categorical_columns_from_s3(bucket_name, s3_client)

    # ✅ Step 2: Compute SHAP values
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Save SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("figures/prelim_shap_summary.png")
    plt.close()

    # Compute SHAP feature importance
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "shap_importance": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="shap_importance", ascending=False)

    selected_features = feature_importance[feature_importance["shap_importance"] > shap_threshold]["feature"].tolist()
    top_features = feature_importance.head(top_k)["feature"].tolist()
    print("Top Features:", top_features)

    # ✅ Select only the features that passed SHAP importance
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()

    # ✅ Identify categorical and numeric features separately
    top_cat_features = [f for f in top_features if f in categorical_columns]
    top_num_features = [f for f in top_features if f not in categorical_columns]

    # ✅ Step 3: Apply Feature Interactions (Without OneHotEncoding)
    interaction_features = []
    new_categorical_features = []  # Track new categorical interactions

    for f1, f2 in combinations(top_features[:5], 2):
        if f1 in top_cat_features and f2 in top_cat_features:
            # ✅ Categorical × Categorical → Concatenation
            new_feature = f"{f1}_x_{f2}"
            X_train_selected[new_feature] = X_train[f1].astype(str) + "_" + X_train[f2].astype(str)
            X_test_selected[new_feature] = X_test[f1].astype(str) + "_" + X_test[f2].astype(str)
            interaction_features.append((f1, f2))
            new_categorical_features.append(new_feature)

        elif f1 in top_num_features and f2 in top_num_features:
            # ✅ Numeric × Numeric → Standard multiplication
            new_feature = f"{f1}_x_{f2}"
            X_train_selected[new_feature] = X_train[f1] * X_train[f2]
            X_test_selected[new_feature] = X_test[f1] * X_test[f2]
            interaction_features.append((f1, f2))

        else:
            # 🚫 **Categorical × Numeric → SKIPPED**
            print(f"Skipping interaction: {f1} × {f2} (Categorical × Numeric)")

    # ✅ Step 4: Encode New Categorical Features & Save Encoding Mappings
    X_train_selected, new_encoders = encode_categorical(X_train_selected, new_categorical_features)
    X_test_selected, _ = encode_categorical(X_test_selected, new_categorical_features)

    # ✅ Step 5: Save transformations for API
    interaction_metadata = {
        "interaction_features": interaction_features,
        "categorical_features": categorical_columns + new_categorical_features,  # Include new categorical features
        "encoders": new_encoders  # Save new encodings
    }

    # Save interaction features
    interaction_file = "figures/interaction_features.json"
    with open(interaction_file, "w") as f:
        json.dump(interaction_metadata, f)

    s3_client.upload_file(interaction_file, bucket_name, "config/interaction_features.json")
    print(f"Interaction features saved to s3://{bucket_name}/config/interaction_features.json")

    return X_train_selected, X_test_selected

def apply_interaction_features(X_new, bucket_name, s3_client):
    """
    Loads transformation metadata from S3 and applies the same interactions for real-time API inference.
    """
    # ✅ Step 1: Load stored interaction transformations
    interaction_key = "config/interaction_features.json"
    interaction_file = "interaction_features.json"
    s3_client.download_file(bucket_name, interaction_key, interaction_file)

    with open(interaction_file, "r") as f:
        interaction_data = json.load(f)

    top_cat_features = interaction_data["categorical_features"]
    interaction_features = interaction_data["interaction_features"]
    encoders = interaction_data["encoders"]

    # ✅ Step 2: Apply interaction transformations
    for f1, f2 in interaction_features:
        new_feature = f"{f1}_x_{f2}"
        if f1 in top_cat_features and f2 in top_cat_features:
            # ✅ Categorical × Categorical → Concatenation
            X_new[new_feature] = X_new[f1].astype(str) + "_" + X_new[f2].astype(str)

        elif f1 not in top_cat_features and f2 not in top_cat_features:
            # ✅ Numeric × Numeric → Standard multiplication
            X_new[new_feature] = X_new[f1] * X_new[f2]

        else:
            # 🚫 **Skipping categorical × numeric interactions**
            print(f"Skipping interaction: {f1} × {f2} (Categorical × Numeric)")

    # ✅ Step 3: Apply Label Encoding using saved mappings
    for col, mapping in encoders.items():
        if col in X_new.columns:
            X_new[col] = X_new[col].map(mapping).fillna(-1).astype(int)  # Map unseen values to -1

    return X_new