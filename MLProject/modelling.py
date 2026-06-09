import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="breast_cancer_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=str, default="None")
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    max_depth = None if args.max_depth == "None" else int(args.max_depth)
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        
        metrics = {"accuracy": acc}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        mlflow.log_artifact("metrics.json")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        
        report = classification_report(y_test, y_pred)
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report.txt')
        
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
