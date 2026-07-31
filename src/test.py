# import pickle, pandas as pd

# with open("models/churn.pkl", "rb") as f:
#     pipeline = pickle.load(f)

# row = pd.DataFrame([{
#     "tenure": 1, "Monthly_Charges": 29.85, "Total_Charges": 29.85,
#     "Is_Married": "Yes", "Dependents": "No", "Paperless_Billing": "Yes",
#     "Streaming_TV": "No", "Streaming_Movies": "No", "Online_Security": "No",
#     "Online_Backup": "Yes", "Device_Protection": "No", "Tech_Support": "No",
#     "Payment_Method": "Electronic check", "Internet_Service": "DSL",
#     "Contract": "Month-to-month"
# }])
# print(pipeline.predict(row), pipeline.predict_proba(row))


import pickle
import pandas as pd

with open("models/churn.pkl", "rb") as f:
    pipeline = pickle.load(f)

customer = {
    "tenure": 34, "Monthly_Charges": 56.95, "Total_Charges": 1889.5,
    "Is_Married": "No", "Dependents": "No", "Paperless_Billing": "No",
    "Streaming_TV": "No", "Streaming_Movies": "No", "Online_Security": "Yes",
    "Online_Backup": "No", "Device_Protection": "Yes", "Tech_Support": "No",
    "Payment_Method": "Mailed check", "Internet_Service": "DSL",
    "Contract": "One year"
}
row = pd.DataFrame([customer])
proba = pipeline.predict_proba(row)[0][1]  # probability of churn

print(f"Churn prediction: {'Yes' if proba >= 0.5 else 'No'} ({proba:.1%} confidence)")


