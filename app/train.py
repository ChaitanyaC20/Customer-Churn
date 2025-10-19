import os
import time
import json
from dotenv import load_dotenv
load_dotenv()
import joblib
import dagshub
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
                             roc_auc_score, RocCurveDisplay, classification_report
)
from xgboost import XGBClassifier

from app.config import (MLFLOW_EXPERIMENT, ARTIFACT_DIR, MODEL_FILE, METRICS_FILE)

dagshub.init(repo_owner='ChaitanyaC20', repo_name='Customer_Churn', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/ChaitanyaC20/Customer_Churn.mlflow")
EXPERIMENT_NAME = MLFLOW_EXPERIMENT or "churn_experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

def train_and_log(X, y, random_state=75):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=80)

    scale_pos_weight = sum(y_train == 0) / max(1, sum(y_train == 1))

    pipe = Pipeline([('scaler', StandardScaler()), ('xg', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
            random_state=80, scale_pos_weight=scale_pos_weight, verbosity=0))])

    param_grid = {
        'xg__n_estimators': [150, 200, 250], 'xg__learning_rate': [0.01, 0.05, 0.1], 'xg__max_depth': [4, 5, 6],
        'xg__colsample_bytree': [0.6, 0.9], 'xg__subsample': [0.3, 0.6, 1.0], 'xg__reg_lambda': [3, 5, 8],
        'xg__reg_alpha': [1, 2, 3]}

    with mlflow.start_run(run_name="XGB_Churn_Model") as run:
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        y_train_probs = best.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_probs >= 0.30).astype(int)

        y_test_probs = best.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_probs >= 0.30).astype(int)

        train_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, average='macro', zero_division=0),
            "train_recall": recall_score(y_train, y_train_pred, average='macro', zero_division=0),
            "train_f1": f1_score(y_train, y_train_pred, average='macro', zero_division=0),
            "train_roc_auc": roc_auc_score(y_train, y_train_probs)
        }
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, average='macro', zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, average='macro', zero_division=0),
            "test_f1": f1_score(y_test, y_test_pred, average='macro', zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_test_probs)
        }
        metrics = {**train_metrics, **test_metrics}

        joblib.dump(best, MODEL_FILE)

        feature_names = X_train.columns.tolist()
        features_path = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")
        joblib.dump(feature_names, features_path)
        mlflow.log_artifact(features_path)

        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        model_path = os.path.join(ARTIFACT_DIR, "xgb_churn_model.joblib")
        joblib.dump(best, model_path)
        mlflow.log_artifact(model_path)

        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()

        RocCurveDisplay.from_predictions(y_test, y_test_probs)
        plt.title("ROC Curve")
        roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches='tight')
        mlflow.log_artifact(roc_path)
        plt.close()

        xgb = best.named_steps['xg']
        importance = pd.Series(xgb.feature_importances_, index=X.columns)
        plt.figure(figsize=(8, 6))
        importance.nlargest(8).plot(kind='barh', color='steelblue')
        plt.title("Top 8 Feature Importances")
        plt.tight_layout()
        fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)
        plt.close()

        report = classification_report(y_test, y_test_pred, output_dict=True)
        report_path = os.path.join(ARTIFACT_DIR, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        comp_path = os.path.join(ARTIFACT_DIR, "train_test_comparison.png")
        comp_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
            "Train": [train_metrics["train_accuracy"], train_metrics["train_precision"],
                      train_metrics["train_recall"], train_metrics["train_f1"], train_metrics["train_roc_auc"]],
            "Test": [test_metrics["test_accuracy"], test_metrics["test_precision"],
                     test_metrics["test_recall"], test_metrics["test_f1"], test_metrics["test_roc_auc"]],})
        comp_df.set_index("Metric").plot(kind="bar", figsize=(7, 5))
        plt.title("Train vs Test Performance Comparison")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(comp_path)
        mlflow.log_artifact(comp_path)
        plt.close()

        # ------------------- ✅ Sample Predictions -------------------
        samples = X_test.sample(5, random_state=random_state)
        sample_probs = best.predict_proba(samples)[:, 1]
        sample_preds = (sample_probs >= 0.30).astype(int)

        examples = pd.DataFrame({
            "Customer_Index": samples.index,
            "Predicted_Label": sample_preds,
            "Churn_Probability": sample_probs
        })
        examples = pd.concat([samples.reset_index(drop=True), examples.reset_index(drop=True)], axis=1)

        # Save predictions to JSON & CSV
        pred_json = os.path.join(ARTIFACT_DIR, "sample_predictions.json")
        pred_csv = os.path.join(ARTIFACT_DIR, "sample_predictions.csv")
        examples.to_json(pred_json, orient="records", indent=2)
        examples.to_csv(pred_csv, index=False)

        mlflow.log_artifact(pred_json)
        mlflow.log_artifact(pred_csv)

        print("\n🎯 Sample Predictions:\n")
        print(examples[["Customer_Index", "Predicted_Label", "Churn_Probability"]])
    return {"model": best, "metrics": metrics}

if __name__ == "__main__":
    from app.data import load_raw_data
    from app.preprocess import preprocess_customer
    df = load_raw_data()
    X, y, feature_columns = preprocess_customer(df)
    results = train_and_log(X, y)
    print("Final Metrics:")
    print(json.dumps(results["metrics"], indent=2))
    time.sleep(2)