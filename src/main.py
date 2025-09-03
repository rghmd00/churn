from load import load_data
from train import train
from evaluate import evaluate



def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data(r'D:\ITI\1_etisalt\Etisalat\churn\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Train and save model
    train(X_train, y_train)

    print("Training complete!")

    evaluate(X_test=X_test,y_test=y_test)





if __name__ == "__main__":
    main()
