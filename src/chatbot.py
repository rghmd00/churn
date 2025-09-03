import os
import asyncio
import pickle
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import initialize_agent, Tool


global_df = 'data/raw/Customer_Churn.csv'



load_dotenv()
st.title("📊 Marketing Assistant with Churn Prediction")

google_api_key = os.getenv("GOOGLE_API_KEY")
chat_model_name = "models/gemini-2.0-flash-lite-preview"

# Ensure event loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --------------------------
# Load your churn model
# --------------------------
with open('models/churn.pkl', "rb") as f:
    churn_model = pickle.load(f)


def predict_churn(data: dict) -> str:
    """
    Takes a dict of customer features and returns churn prediction.
    Expected fields:
    CustomerId, Gender, Senior_Citizen, Is_Married, Dependents, tenure,
    Phone_Service, Dual, Internet_Service, Online_Security, Online_Backup,
    Device_Protection, Tech_Support, Streaming_TV, Streaming_Movies,
    Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges
    """
    try:
        # Expected schema
        expected_cols = [
            "CustomerId", "Gender", "Senior_Citizen", "Is_Married", "Dependents",
            "tenure", "Phone_Service", "Dual", "Internet_Service",
            "Online_Security", "Online_Backup", "Device_Protection", "Tech_Support",
            "Streaming_TV", "Streaming_Movies", "Contract", "Paperless_Billing",
            "Payment_Method", "Monthly_Charges", "Total_Charges"
        ]

        one_raw = pd.DataFrame([data])

        # Ensure all expected columns exist
        for col in expected_cols:
            if col not in one_raw.columns:
                one_raw[col] = None   # placeholder

        # Fix numeric columns
        for col in ["tenure", "Monthly_Charges", "Total_Charges"]:
            one_raw[col] = pd.to_numeric(one_raw[col], errors="coerce").fillna(0)

        # Predict
        pred = churn_model.predict(one_raw)[0]
        prob = None
        if hasattr(churn_model, "predict_proba"):
            prob = churn_model.predict_proba(one_raw)[0].max()

        # Map back 0/1 to Yes/No
        target_map = {0: "No", 1: "Yes"}
        pred_label = target_map.get(pred, pred)

        if prob:
            return f"Prediction: {pred_label} (confidence: {prob:.2f})"
        return f"Prediction: {pred_label}"

    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"



def summary_stats(df) -> str:
    try:
        if isinstance(df, str) and df.endswith(".csv"):
            df = pd.read_csv(df)
        elif isinstance(df, str):
            import json
            df = pd.DataFrame(json.loads(df))
        
        return "\n" + df.describe(include="all").to_string()
    except Exception as e:
        return f"❌ Error in summary_stats: {str(e)}"

################################## TOOLS ############################################
stats_tool = Tool(
    name="Summary Stats",
    func=lambda _: summary_stats(global_df),
    description="Returns descriptive statistics for numerical and categorical features."
)

churn_predic_tool = Tool(
        name="Churn Predictor",
        func=predict_churn,
        description=(
            "Use this tool to predict if a customer will churn. "
            "Provide customer details as a JSON dict with keys such as "
            "Gender, Senior_Citizen, Is_Married, Dependents, tenure, "
            "Phone_Service, Internet_Service, Contract, Monthly_Charges, "
            "Total_Charges, etc."
        )
    )



# Wrap model as tool
tools = [
    churn_predic_tool,
    stats_tool
]

# --------------------------
# Session state for messages
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage(
        "You are a marketing assistant. "
        "If the user asks about churn prediction, use the Churn Predictor tool. "
        "Otherwise, answer normally,also When using tools like summary_stats, "
            "always include the raw output of the tool in your final answer."
    ))

# --------------------------
# Display previous messages
# --------------------------
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --------------------------
# Chat input
# --------------------------
prompt = st.chat_input("Ask about churn or marketing insights...")

if prompt:
    # Add user input
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # Create LLM + Agent
    model_llm = ChatGoogleGenerativeAI(
        model=chat_model_name,
        temperature=0,
        google_api_key=google_api_key
    )

    agent = initialize_agent(
        tools=tools,
        llm=model_llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True

    )

    # Run agent
    result = agent.run(prompt)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append(AIMessage(result))
