import os
import asyncio
import pickle
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain integrations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import initialize_agent, Tool

import json

# --------------------------
# Setup
# --------------------------
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
with open("models/churn.pkl", "rb") as f:
    churn_model = pickle.load(f)

# --------------------------
# Prediction function
# --------------------------
def predict_churn(data):
    try:
        # Parse input
        if isinstance(data, str):
            data = json.loads(data)

        EXPECTED_COLS = [
            'Senior_Citizen', 'Is_Married', 'Dependents', 'tenure',
            'Internet_Service', 'Online_Security', 'Online_Backup',
            'Device_Protection', 'Tech_Support', 'Streaming_TV',
            'Streaming_Movies', 'Contract', 'Paperless_Billing',
            'Payment_Method', 'Monthly_Charges', 'Total_Charges'
        ]

        df = pd.DataFrame([data])
        df = df[EXPECTED_COLS]

        # Fix numeric fields
        for col in ['Senior_Citizen', 'tenure', 'Monthly_Charges', 'Total_Charges']:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Prediction
        pred = churn_model.predict(df)[0]
        proba = churn_model.predict_proba(df)[0] if hasattr(churn_model, "predict_proba") else None

        target_map = {0: "No", 1: "Yes"}
        pred_label = target_map.get(pred, str(pred))

        return {
            "prediction": pred_label,
            "probabilities": {
                "No": float(proba[0]) if proba is not None else None,
                "Yes": float(proba[1]) if proba is not None else None,
            },
            "confidence": float(proba[pred]) if proba is not None else None,
            "inputs": data
        }

    except Exception as e:
        return {"error": str(e)}

# --------------------------
# Wrap model as tool
# --------------------------
tools = [
    Tool(
        name="Churn Predictor",
        func=predict_churn,
        description=(
            "Predict if a customer will churn. "
            "Provide details as a JSON dict with keys: "
            "'Senior_Citizen', 'Is_Married', 'Dependents', 'tenure', "
            "'Internet_Service', 'Online_Security', 'Online_Backup', "
            "'Device_Protection', 'Tech_Support', 'Streaming_TV', "
            "'Streaming_Movies', 'Contract', 'Paperless_Billing', "
            "'Payment_Method', 'Monthly_Charges', 'Total_Charges'."
        )
    )
]

# --------------------------
# Session state for messages
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage(
        "You are a marketing assistant. "
        "If the user asks about churn prediction, use the Churn Predictor tool "
        "and return a full churn report. Otherwise, answer normally."
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
        if isinstance(result, dict) and "prediction" in result:
            st.subheader("🔮 Churn Prediction Report")
            st.write(f"**Prediction:** {result['prediction']}")
            st.write(f"**Confidence:** {result['confidence']:.2f}" if result["confidence"] else "N/A")

            # Probability breakdown
            st.write("### Probability Breakdown")
            st.table(pd.DataFrame([result["probabilities"]]))

            # Customer inputs
            st.write("### Customer Details")
            st.json(result["inputs"])

            # Bar chart
            if result["probabilities"]["No"] is not None:
                st.bar_chart(result["probabilities"])
        else:
            st.markdown(result)

    st.session_state.messages.append(AIMessage(str(result)))
