

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier


def coerce_numeric(df):
    """Safely converts string numbers/blanks to numeric floats and NaNs."""
    return df.apply(pd.to_numeric, errors='coerce')


def train(X_train, y_train, model_path='models/churn.pkl'):
    X_train = X_train.copy()

    numeric_cols = ['tenure', 'Monthly_Charges', 'Total_Charges']
    binary_cols = ['Is_Married', 'Dependents', 'Paperless_Billing']
    service_cols = ["Streaming_TV", "Streaming_Movies", "Online_Security",
                    "Online_Backup", "Device_Protection", "Tech_Support"]
    nominal_cols = ['Payment_Method', 'Internet_Service', 'Contract']

    # 1. Numeric pipeline with self-contained type coercion
    numeric_pipeline = Pipeline([
        ("to_numeric", FunctionTransformer(coerce_numeric)),
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # 2. Categorical pipelines
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    service_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_cols),
        ('binary', binary_pipeline, binary_cols),
        ('service', service_pipeline, service_cols),
        ('nominal', nominal_pipeline, nominal_cols),
    ])

    # 3. Class imbalance dampening
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    raw_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
    scale_pos_weight = raw_ratio ** 0.5

    print(f"Training data: {n_pos} positive, {n_neg} negative (raw ratio: {raw_ratio:.2f})")
    print(f"Using scale_pos_weight={scale_pos_weight:.4f}")

    # 4. Regularized XGBoost configuration to prevent overfitting
    classifier = XGBClassifier(
        n_estimators=180,             # Controlled number of trees (safe for full dataset training)
        learning_rate=0.04,           # Slower learning rate for smoother convergence
        max_depth=3,                  # Shallow trees to eliminate hyper-specific leaf rules
        min_child_weight=5,           # Requires at least 5 instances per node split
        subsample=0.8,                # Train each tree on 80% random rows (bagging)
        colsample_bytree=0.8,         # Select 80% random features per tree
        reg_alpha=0.1,                # L1 regularization on leaf weights
        reg_lambda=2.0,               # L2 regularization on leaf weights
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

    # 5. Build and fit the unified pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    pipeline.fit(X_train, y_train)

    # 6. Evaluation metrics on training set
    train_preds = pipeline.predict(X_train)
    train_probs = pipeline.predict_proba(X_train)[:, 1]
    
    print("\n--- Model Training Summary ---")
    print(f"Train AUC: {roc_auc_score(y_train, train_probs):.4f}")
    print(classification_report(y_train, train_preds))

    # 7. Persist pipeline
    dirname = os.path.dirname(model_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"Model successfully saved to: {model_path}")
    return pipeline