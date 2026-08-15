# Churn Prediction Marketing Assistant

A chat-based marketing assistant that predicts customer churn using a trained XGBoost model, with a local LLM (via Ollama) extracting customer details from natural-language messages.

## What it does

- User describes a customer in plain English (e.g. *"3 months tenure, $95/month, month-to-month contract..."*)
- A local LLM (`qwen2.5:3b-instruct`) extracts structured fields and calls a `predict_churn` tool
- The tool runs the customer data through a trained sklearn/XGBoost pipeline and returns a churn prediction with probabilities
- Streamlit renders a chat interface and a visual prediction report

## Project structure
src/
main.py # train() and evaluate() — model training pipeline
chat.py # Streamlit chat app with LLM + churn prediction tool
test.py # manual testing script (bypasses the LLM)
models/
churn.pkl # pickled sklearn Pipeline (preprocessing + XGBClassifier)


## How it works

**Training (`main.py`)**
- Preprocesses numeric, binary, service, and nominal columns via a `ColumnTransformer` (median/mode imputation + ordinal/one-hot encoding)
- Corrects class imbalance using a dampened `scale_pos_weight` (sqrt of the positive/negative ratio)
- Trains an `XGBClassifier`, saves the fitted pipeline to `models/churn.pkl`

**Chat app (`chat.py`)**
- LLM extracts customer fields from free text into a `predict_churn` tool call
- Missing fields are left as `None` and handled automatically by the pipeline's imputers — the user isn't required to provide every field
- Results (prediction, confidence, probabilities, count of missing fields) are shown in a report card