import streamlit as st
import pandas as pd

st.set_page_config(page_title="Churn Prediction Guide", layout="centered")

st.title("How to Ask About Churn Prediction")
st.write(
    "You can submit customer information in **natural language** or as **JSON**. "
    "Unspecified fields will automatically use baseline model defaults."
)

st.info(
    "Tip for best results: Clearly state the attribute names (for example, 'Contract: Month-to-month' "
    "or 'Tenure: 12 months') to ensure accurate feature extraction."
)

tab_nl, tab_json, tab_ref = st.tabs(["Natural Language", "JSON Format", "Field Reference"])

# ---------------------------------------------------------
# Tab 1: Natural Language Example
# ---------------------------------------------------------
with tab_nl:
    st.caption("Paste into chat or use as a template:")
    nl_example = """Check churn risk for this customer:
- Tenure: 5 months
- Contract: Month-to-month
- Monthly Charges: 75
- Internet Service: Fiber optic
- Tech Support: No
- Payment Method: Electronic check"""
    st.code(nl_example, language="markdown")
    st.caption("Note: Using a clear list format ensures reliable entity parsing.")

# ---------------------------------------------------------
# Tab 2: JSON Payload Example
# ---------------------------------------------------------
with tab_json:
    st.caption("Structured key-value format:")
    json_example = """{
  "tenure": 5,
  "Contract": "Month-to-month",
  "Monthly_Charges": 75.0,
  "Total_Charges": 350.0,
  "Internet_Service": "Fiber optic",
  "Tech_Support": "No",
  "Payment_Method": "Electronic check",
  "Paperless_Billing": "Yes"
}"""
    st.code(json_example, language="json")

# ---------------------------------------------------------
# Tab 3: Grouped Field Reference Table
# ---------------------------------------------------------
with tab_ref:
    st.caption("Supported attributes and allowed values:")
    
    reference_data = [
        {
            "Category": "Core Drivers",
            "Fields": "Contract, tenure, Monthly_Charges, Total_Charges",
            "Allowed / Typical Values": "Month-to-month / One year / Two year, Integer months (e.g. 5), Numeric values"
        },
        {
            "Category": "Demographics",
            "Fields": "Senior_Citizen, Is_Married, Dependents",
            "Allowed / Typical Values": "Yes / No (or 1 / 0)"
        },
        {
            "Category": "Connectivity",
            "Fields": "Internet_Service",
            "Allowed / Typical Values": "DSL, Fiber optic, No"
        },
        {
            "Category": "Add-on Services",
            "Fields": "Online_Security, Tech_Support, Online_Backup, Device_Protection, Streaming_TV, Streaming_Movies",
            "Allowed / Typical Values": "Yes / No"
        },
        {
            "Category": "Billing",
            "Fields": "Payment_Method, Paperless_Billing",
            "Allowed / Typical Values": "Electronic check, Mailed check, Bank transfer, Credit card | Yes / No"
        }
    ]
    
    st.dataframe(pd.DataFrame(reference_data), hide_index=True, width='stretch')