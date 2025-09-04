import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline  
from imblearn.over_sampling import SMOTE


def train(X_train, y_train):

    binary_cols = ['Is_Married', 'Dependents', 'Paperless_Billing']
    service_cols = ["Streaming_TV", "Streaming_Movies", "Online_Security",
                    "Online_Backup", "Device_Protection", "Tech_Support"]
    ordinal_cols = ['Payment_Method', 'Internet_Service', 'Contract']
    numeric_cols = ['tenure', 'Monthly_Charges', 'Total_Charges']

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_train[col] = X_train[col].fillna(0)

    # Define encoders with fixed categories
    payment_categories = [["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]]
    payment_encoder = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=payment_categories, handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    internet_encoder = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[["No", "DSL", "Fiber optic"]], handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    contract_encoder = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]], handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # Build ordinal pipeline
    ordinal_pipeline = ColumnTransformer([
        ('payment', payment_encoder, ['Payment_Method']),
        ('internet', internet_encoder, ['Internet_Service']),
        ('contract', contract_encoder, ['Contract'])
    ])

    # Preprocessor with imputers for all categorical cols
    preprocessor = ColumnTransformer(transformers=[
        ('binary', Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), binary_cols),

        ('service', Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), service_cols),

        ('ordinal', ordinal_pipeline, ordinal_cols),
    ])

    # Use imblearn Pipeline to include SMOTE
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier())
    ])

    pipeline.fit(X_train, y_train)

    with open('models/churn.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print("Model saved as churn.pkl")













# import os
# import pickle
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.ensemble import GradientBoostingClassifier
# from imblearn.pipeline import Pipeline  
# from imblearn.over_sampling import SMOTE


# def train(X_train, y_train):

#     binary_cols = ['Is_Married', 'Dependents', 'Paperless_Billing']
#     service_cols = ["Streaming_TV", "Streaming_Movies", "Online_Security",
#                     "Online_Backup", "Device_Protection", "Tech_Support"]
#     ordinal_cols = ['Payment_Method', 'Internet_Service', 'Contract']
#     numeric_cols = ['tenure', 'Monthly_Charges', 'Total_Charges']

#     for col in numeric_cols:
#         X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
#         X_train[col] = X_train[col].fillna(0)

#     # Define encoders
#     payment_categories = [["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]]
#     payment_encoder = OrdinalEncoder(categories=payment_categories)
#     internet_encoder = OrdinalEncoder(categories=[["No", "DSL", "Fiber optic"]])
#     contract_encoder = OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]])

#     ordinal_pipeline = ColumnTransformer([
#         ('payment', payment_encoder, ['Payment_Method']),
#         ('internet', internet_encoder, ['Internet_Service']),
#         ('contract', contract_encoder, ['Contract'])
#     ])

#     preprocessor = ColumnTransformer(transformers=[
#         ('binary', OrdinalEncoder(), binary_cols),
#         ('service', OrdinalEncoder(), service_cols),
#         ('ordinal', ordinal_pipeline, ordinal_cols),
#     ])

#     # Use imblearn Pipeline to include SMOTE
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         # ('smote', SMOTE(random_state=42)),
#         ('classifier', GradientBoostingClassifier())
#     ])

    

#     pipeline.fit(X_train, y_train)

#     with open('models/churn.pkl', 'wb') as f:
#         pickle.dump(pipeline, f)

#     print("Model saved as churn_model.pkl")
    

