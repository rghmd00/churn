import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(path:str):

    df = pd.read_csv(path)
        
    # y = df["Churn"]
    # target_encoder = LabelEncoder()
    # y = target_encoder.fit_transform(y) 
    # Manual mapping

    y = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop(columns=["Churn", "customerID","gender",'Dual','Phone_Service'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test



