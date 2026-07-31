import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier


def train(X_train, y_train, model_path='models/churn.pkl'):
    # Work on a copy so we don't mutate the caller's DataFrame
    X_train = X_train.copy()

    numeric_cols = ['tenure', 'Monthly_Charges', 'Total_Charges']
    binary_cols = ['Is_Married', 'Dependents', 'Paperless_Billing']
    service_cols = ["Streaming_TV", "Streaming_Movies", "Online_Security",
                     "Online_Backup", "Device_Protection", "Tech_Support"]
    nominal_cols = ['Payment_Method', 'Internet_Service', 'Contract']

    # Coerce numeric columns; leave NaNs for the imputer to handle properly
    X_train[numeric_cols] = X_train[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Numeric pipeline: median impute (don't silently zero-fill)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Binary/ordinal-ish categoricals
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # Service columns (also effectively categorical, low cardinality)
    service_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # Nominal columns with no inherent order -> one-hot instead of ordinal
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

    # Train/validation split for early stopping + evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()   
    raw_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0

    # Soften the imbalance correction — full ratio overcorrects for recall
    # at a steep cost to precision (roughly half of "Yes" predictions were
    # false positives at the raw ratio). sqrt dampens this while still
    # giving churners meaningfully more weight than a ratio of 1.0 would.
    scale_pos_weight = raw_ratio ** 0.5

    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
    )

    preprocessor.fit(X_tr)
    X_tr_enc = preprocessor.transform(X_tr)
    X_val_enc = preprocessor.transform(X_val)

    classifier.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_val_enc, y_val)],
        verbose=False
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    val_preds = classifier.predict(X_val_enc)
    val_probs = classifier.predict_proba(X_val_enc)[:, 1]
    print(f"Validation AUC: {roc_auc_score(y_val, val_probs):.4f}")
    print(classification_report(y_val, val_preds))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"Model saved as {model_path}")
    return pipeline