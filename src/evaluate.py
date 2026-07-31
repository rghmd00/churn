
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix


def evaluate(X_test, y_test, model_path='models/churn.pkl'):
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    # Must match the numeric coercion done in train() —
    # this step lives outside the pickled pipeline, so it has
    # to be replicated here or predict() will choke on raw strings.
    X_test = X_test.copy()
    numeric_cols = ['tenure', 'Monthly_Charges', 'Total_Charges']
    X_test[numeric_cols] = X_test[numeric_cols].apply(pd.to_numeric, errors='coerce')

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))

    
    # after evaluate() computes preds
    print(confusion_matrix(y_test, preds))

    return preds, 

