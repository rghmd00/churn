import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
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

    # Define encoders
    payment_categories = [["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]]
    payment_encoder = OrdinalEncoder(categories=payment_categories)
    internet_encoder = OrdinalEncoder(categories=[["No", "DSL", "Fiber optic"]])
    contract_encoder = OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]])

    ordinal_pipeline = ColumnTransformer([
        ('payment', payment_encoder, ['Payment_Method']),
        ('internet', internet_encoder, ['Internet_Service']),
        ('contract', contract_encoder, ['Contract'])
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('binary', OrdinalEncoder(), binary_cols),
        ('service', OrdinalEncoder(), service_cols),
        ('ordinal', ordinal_pipeline, ordinal_cols),
    ])

    # Use imblearn Pipeline to include SMOTE
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier())
    ])

    

    pipeline.fit(X_train, y_train)

    with open('models/churn.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print("Model saved as churn_model.pkl")
    




# import pickle


# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.ensemble import GradientBoostingClassifier
# import pandas as pd

# def train(X_train,y_train):

#     binary_cols = ['Is_Married', 'Dependents', 'Paperless_Billing']
#     service_cols = ["Streaming_TV", "Streaming_Movies", "Online_Security","Online_Backup", "Device_Protection", "Tech_Support"]
#     ordinal_cols = ['Payment_Method', 'Internet_Service', 'Contract']
#     numeric_cols = ['tenure', 'Monthly_Charges', 'Total_Charges']  



#     for col in numeric_cols:
#         X_train[col] = pd.to_numeric(X_train[col], errors='coerce') 
#         X_train[col] = X_train[col].fillna(0)                       


#     # df["Total_Charges"] = pd.to_numeric(df["Total_Charges"], errors="coerce")
#     # df["Total_Charges"] = df["Total_Charges"].fillna(0)



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
#         # ('numeric', StandardScaler(), numeric_cols)
#     ])


#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', GradientBoostingClassifier())
#     ])



#     # Train
#     pipeline.fit(X_train, y_train)


#     with open(r'D:\ITI\1_etisalt\Etisalat\churn\models\churn.pkl', 'wb') as f:
#         pickle.dump(pipeline, f)

#     print("Model saved as churn_model.pkl")
