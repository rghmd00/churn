import pickle
from sklearn.metrics import accuracy_score, classification_report


def evaluate(X_test,y_test):

    with open('models/churn.pkl', 'rb') as f:
        pipeline = pickle.load(f)

        
    preds = pipeline.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))