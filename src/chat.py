# uv run python -m streamlit run .\src\chat.py
import pickle
import pandas as pd
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import Optional, Union

st.title("Marketing Assistant with Churn Prediction")

with open("models/churn.pkl", "rb") as f:
    churn_model = pickle.load(f)


@tool
def predict_churn(tenure: Optional[int] = None, Monthly_Charges: Optional[float] = None,
                   Total_Charges: Optional[float] = None,
                   Is_Married: Optional[Union[str, bool]] = None,
                   Dependents: Optional[Union[str, bool]] = None,
                   Paperless_Billing: Optional[Union[str, bool]] = None,
                   Streaming_TV: Optional[Union[str, bool]] = None,
                   Streaming_Movies: Optional[Union[str, bool]] = None,
                   Online_Security: Optional[Union[str, bool]] = None,
                   Online_Backup: Optional[Union[str, bool]] = None,
                   Device_Protection: Optional[Union[str, bool]] = None,
                   Tech_Support: Optional[Union[str, bool]] = None,
                   Payment_Method: Optional[str] = None,
                   Internet_Service: Optional[str] = None,
                   Contract: Optional[str] = None) -> dict:
    """Predict whether a customer will churn. Leave any unknown field as None."""
    args = locals()

    yes_no_fields = ["Is_Married", "Dependents", "Paperless_Billing",
                      "Streaming_TV", "Streaming_Movies", "Online_Security",
                      "Online_Backup", "Device_Protection", "Tech_Support"]
    for field in yes_no_fields:
        v = args[field]
        if isinstance(v, bool):
            args[field] = "Yes" if v else "No"

    row = pd.DataFrame([args])
    numeric_cols = ['tenure', 'Monthly_Charges', 'Total_Charges']
    row[numeric_cols] = row[numeric_cols].apply(pd.to_numeric, errors='coerce')
    n_missing = sum(1 for v in args.values() if v is None)

    proba = churn_model.predict_proba(row)[0]
    return {
        "prediction": "Yes" if proba[1] >= 0.5 else "No",
        "probabilities": {"No": float(proba[0]), "Yes": float(proba[1])},
        "confidence": float(max(proba)),
        "fields_missing": n_missing,
    }


def render_report(result: dict):
    st.subheader("Churn Prediction Report")
    if result.get("fields_missing", 0) > 0:
        st.warning(f"⚠️ {result['fields_missing']} field(s) were missing and estimated automatically — prediction may be less reliable.")
    st.write(f"**Prediction:** {result['prediction']}")
    st.write(f"**Confidence:** {result['confidence']:.2%}")
    st.bar_chart(result["probabilities"])


if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(
        "You are a marketing assistant. When the user describes a customer, "
        "call predict_churn immediately, extracting every field the user "
        "mentioned. Leave any field the user did not mention as None — "
        "do not ask the user for missing fields, the tool handles them "
        "automatically. Only ask a clarifying question if the user's message "
        "is unrelated to a specific customer."
    )]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").markdown(msg.content)

if prompt := st.chat_input("Ask about churn or marketing insights..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt)) #type: ignore

    llm = ChatOllama(model="qwen2.5:3b-instruct", temperature=0).bind_tools([predict_churn])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(reply) #type: ignore

        if reply.tool_calls: #type: ignore
            call = reply.tool_calls[0] #type: ignore
            st.expander("Debug: extracted fields").json(call["args"])
            result = predict_churn.invoke(call["args"])
            st.session_state.messages.append(ToolMessage(str(result), tool_call_id=call["id"])) #type: ignore

            if "error" not in result:
                render_report(result)

            summary = llm.invoke(st.session_state.messages)
            st.markdown(summary.content)
            st.session_state.messages.append(summary) #type: ignore
        else:
            st.markdown(reply.content)


# SystemMessage(
#     "You are a marketing assistant. When predicting churn, extract EVERY "
#     "field mentioned in the user's message — do not leave a field as None "
#     "if the user stated it, even implicitly (e.g. 'been with us for 3 months' "
#     "means tenure=3). Example: 'married with no kids, 2yr contract, $60/mo' "
#     "-> Is_Married='Yes', Dependents='No', Contract='Two year', Monthly_Charges=60. "
#     "Ask the user for any field you genuinely cannot determine before calling the tool."
#     )