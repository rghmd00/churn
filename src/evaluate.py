import pickle
from sklearn.metrics import accuracy_score, classification_report



def evaluate(X_test,y_test):

    with open(r'D:\ITI\1_etisalt\Etisalat\churn\models\churn.pkl', 'rb') as f:
        loaded_pipeline = pickle.load(f)

    preds = loaded_pipeline.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
