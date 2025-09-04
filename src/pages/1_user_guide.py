import streamlit as st

st.set_page_config(page_title="📘 Churn Prediction Guide", layout="centered")

st.title("📘 How to Ask About Churn Prediction")
st.write(
    "This assistant predicts whether a customer is at risk of churn. "
    "You can provide details in **natural language** or as **JSON**. "
    "Use the examples below — each block has a copy button."
)

st.divider()

tab1, tab2, tab3 = st.tabs(["🗣️ Natural Language Example", "🧱 JSON Example", "📚 Field Reference"])

with tab1:
    st.subheader("Natural language prompt (copy & paste)")
    nl_example = """Can you check if this customer is at risk of churn?

Senior citizen: No
Married: Yes
Dependents: No
Tenure: 5 months
Internet service: DSL
Online security: No
Online backup: Yes
Device protection: No
Tech support: No
Streaming TV: Yes
Streaming movies: No
Contract: Month-to-month
Paperless billing: Yes
Payment method: Electronic check
Monthly charges: 75
Total charges: 350
"""
    st.code(nl_example, language="markdown")
    st.caption("Tip: Natural language is fine — the app will parse these values into the model format.")

with tab2:
    st.subheader("JSON payload (for advanced users)")
    json_example = """{
  "Senior_Citizen": 0,
  "Is_Married": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "Internet_Service": "DSL",
  "Online_Security": "No",
  "Online_Backup": "Yes",
  "Device_Protection": "No",
  "Tech_Support": "No",
  "Streaming_TV": "Yes",
  "Streaming_Movies": "No",
  "Contract": "Month-to-month",
  "Paperless_Billing": "Yes",
  "Payment_Method": "Electronic check",
  "Monthly_Charges": 75,
  "Total_Charges": 350
}"""
    st.code(json_example, language="json")
    st.caption("Exactly matches the model’s expected column names and value formats.")

with tab3:
    st.subheader("Supported Fields & Allowed Values")
    st.markdown(
        """
- **Senior_Citizen**: `0` or `1` *(or “No/Yes” in natural language)*  
- **Is_Married**: `Yes` / `No`  
- **Dependents**: `Yes` / `No`  
- **tenure**: Integer number of months (e.g., `5`)  
- **Internet_Service**: `No` / `DSL` / `Fiber optic`  
- **Online_Security**: `Yes` / `No`  
- **Online_Backup**: `Yes` / `No`  
- **Device_Protection**: `Yes` / `No`  
- **Tech_Support**: `Yes` / `No`  
- **Streaming_TV**: `Yes` / `No`  
- **Streaming_Movies**: `Yes` / `No`  
- **Contract**: `Month-to-month` / `One year` / `Two year`  
- **Paperless_Billing**: `Yes` / `No`  
- **Payment_Method**: `Electronic check` / `Mailed check` / `Bank transfer (automatic)` / `Credit card (automatic)`  
- **Monthly_Charges**: Numeric (e.g., `75` or `75.0`)  
- **Total_Charges**: Numeric (e.g., `350` or `350.0`)  
"""
    )

st.divider()
st.info(
    "✅ You can use either format. The main assistant page will parse your input and return the prediction and confidence."
)
