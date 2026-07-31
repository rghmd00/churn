
# uv run python -m streamlit run .\src\chat.py
import pickle
import pandas as pd
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import Optional

st.title("Marketing Assistant with Churn Prediction")

with open("models/churn.pkl", "rb") as f:
    churn_model = pickle.load(f)


@tool
def predict_churn(tenure: Optional[int] = None, Monthly_Charges: Optional[float] = None,
                   Total_Charges: Optional[float] = None, Is_Married: Optional[str] = None,
                   Dependents: Optional[str] = None, Paperless_Billing: Optional[str] = None,
                   Streaming_TV: Optional[str] = None, Streaming_Movies: Optional[str] = None,
                   Online_Security: Optional[str] = None, Online_Backup: Optional[str] = None,
                   Device_Protection: Optional[str] = None, Tech_Support: Optional[str] = None,
                   Payment_Method: Optional[str] = None, Internet_Service: Optional[str] = None,
                   Contract: Optional[str] = None) -> dict:
    """Predict whether a customer will churn. Yes/No fields must be the
    exact string "Yes" or "No" (not true/false). Call this only once you
    have values for every field."""
    args = locals()
    missing = [k for k, v in args.items() if v is None]
    if missing:
        return {"error": f"Missing required fields: {missing}. Ask the user for these before retrying."}

    row = pd.DataFrame([args])
    proba = churn_model.predict_proba(row)[0]
    return {
        "prediction": "Yes" if proba[1] >= 0.5 else "No",
        "probabilities": {"No": float(proba[0]), "Yes": float(proba[1])},
        "confidence": float(max(proba)),
    }


def render_report(result: dict):
    st.subheader("Churn Prediction Report")
    st.write(f"**Prediction:** {result['prediction']}")
    st.write(f"**Confidence:** {result['confidence']:.2%}")
    st.bar_chart(result["probabilities"])


if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(
        "You are a marketing assistant. Use predict_churn for churn questions; "
        "ask for any missing fields first."
    )]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").markdown(msg.content)

if prompt := st.chat_input("Ask about churn or marketing insights..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    llm = ChatOllama(model="qwen2.5:3b-instruct", temperature=0).bind_tools([predict_churn])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(reply)

        if reply.tool_calls:
            call = reply.tool_calls[0]
            result = predict_churn.invoke(call["args"])
            st.session_state.messages.append(ToolMessage(str(result), tool_call_id=call["id"]))

            if "error" not in result:
                render_report(result)

            summary = llm.invoke(st.session_state.messages)
            st.markdown(summary.content)
            st.session_state.messages.append(summary)
        else:
            st.markdown(reply.content)