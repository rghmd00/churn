
import pickle
import pandas as pd

with open("models/churn.pkl", "rb") as f:
    pipeline = pickle.load(f)

customer_1 = {
    "tenure": 2, "Monthly_Charges": 90, "Total_Charges": 180,
    "Is_Married": "No", "Dependents": "No", "Paperless_Billing": "Yes",
    "Streaming_TV": "No", "Streaming_Movies": "No", "Online_Security": "No",
    "Online_Backup": "No", "Device_Protection": "No", "Tech_Support": "No",
    "Payment_Method": "Electronic check", "Internet_Service": "Fiber optic",
    "Contract": "Month-to-month"
}
row = pd.DataFrame([customer_1])
proba = pipeline.predict_proba(row)[0][1]  # probability of churn

prediction = "Yes" if proba >= 0.5 else "No"
confidence = proba if prediction == "Yes" else 1 - proba
print(f"Churn prediction: {prediction} ({confidence:.1%} confidence)")

