import argparse
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

def main(args):
    # Load data
    df = pd.read_csv(args.training_data)

    X = df.drop("fraud_reported", axis=1)
    y = df["fraud_reported"].map({"Y":1, "N":0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save model
    #mlflow.xgboost.save_model(model, args.model_output)
  
    mlflow.xgboost.log_model(
    xgboost_model=model,
    artifact_path="insurancefrauddet_xgboost_model", # This name will be the folder name in your artifacts
    registered_model_name="claimsfraud_xgboost_model" # This registers it in the Model registry
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str)
    #parser.add_argument("--model_output", type=str)
    args = parser.parse_args()
    main(args)
