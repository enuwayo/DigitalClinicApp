# src/predict/inference.py

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from Education.src.data.make_dataset import preprocess_data
from Education.src.features.build_features import split_features_labels

def load_model(model_path='Education/src/models/xgb_model.pkl'):
    return joblib.load(model_path)

def evaluate_on_unseen(model, df: pd.DataFrame):
    df = preprocess_data(df)
    X, y_true = split_features_labels(df)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }

    df["Predicted_Label"] = predictions
    return df, metrics