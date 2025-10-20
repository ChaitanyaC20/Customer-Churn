import os
import time
import json
from dotenv import load_dotenv
load_dotenv()
import joblib
import dagshub
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from app.config import (MLFLOW_EXPERIMENT_Reg, ARTIFACT_DIR_REG, MODEL_FILE_REG, METRICS_FILE_REG)

dagshub.init(repo_owner='ChaitanyaC20', repo_name='Total_Revenue_Regression', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/ChaitanyaC20/Total_Revenue_Regression.mlflow")
EXPERIMENT_NAME = MLFLOW_EXPERIMENT_Reg or "Regression"
print("Tracking URI:", mlflow.get_tracking_uri())
mlflow.set_experiment(EXPERIMENT_NAME)

def train_and_log_reg(X, y, random_state=75):
    os.makedirs(ARTIFACT_DIR_REG, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle= True, test_size=0.25, random_state=42)

    pipe = Pipeline([('scaler', StandardScaler()),('lg', LGBMRegressor())])

    param_grid = {'lg__num_leaves': [20, 30], 'lg__learning_rate': [0.01, 0.02], 'lg__n_estimators': [100, 200],
        'lg__reg_lambda': [2.0, 3.0], 'lg__reg_alpha': [0.1, 0.5, 1.0], 'lg__subsample': [0.9, 1.0], 'lg__colsample_bytree': [0.9, 1.0],}

    with mlflow.start_run(run_name="lg_Churn_Reg") as run:
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        y_train_pred = best.predict(X_train)
        y_test_pred = best.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        cv_score = cross_val_score(best, X, y, cv=5, scoring="r2").mean()

        metrics = {
            "cv_r2_mean": cv_score,
            "train_rmse": train_rmse, "test_rmse": test_rmse,
            "train_mae": train_mae, "test_mae": test_mae,
            "train_mse": train_mse, "test_mse": test_mse,
            "train_r2": train_r2, "test_r2": test_r2
        }

        joblib.dump(best, MODEL_FILE_REG)
        model_path = os.path.join(ARTIFACT_DIR_REG, "lgbm_revenue_model.joblib")
        joblib.dump(best, model_path)
        mlflow.log_artifact(model_path)

        feature_names = X_train.columns.tolist()
        features_path = os.path.join(ARTIFACT_DIR_REG, "feature_columns.pkl")
        joblib.dump(feature_names, features_path)
        mlflow.log_artifact(features_path)

        # Log params + metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        with open(METRICS_FILE_REG, "w") as f:
            json.dump(metrics, f, indent=2)

        lg = best.named_steps['lg']
        importance = pd.Series(lg.feature_importances_, index=X.columns)
        plt.figure(figsize=(8, 6))
        importance.nlargest(10).plot(kind='barh', color='steelblue')
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        fi_path = os.path.join(ARTIFACT_DIR_REG, "feature_importance.png")
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Revenue')
        plt.ylabel('Predicted Revenue')
        plt.title('Actual vs Predicted Total Revenue')
        plt.grid(True)
        actual_vs_pred_path = os.path.join(ARTIFACT_DIR_REG, "actual_vs_predicted.png")
        plt.savefig(actual_vs_pred_path)
        mlflow.log_artifact(actual_vs_pred_path)
        plt.close()

        lg = best.named_steps['lg']
        importance = pd.Series(lg.feature_importances_, index=X.columns)
        plt.figure(figsize=(8, 6))
        importance.nlargest(8).plot(kind='barh', color='steelblue')
        plt.title("Top 8 Feature Importances")
        plt.tight_layout()
        fi_path = os.path.join(ARTIFACT_DIR_REG, "feature_importance.png")
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)
        plt.close()

        print(f"✅ Run logged successfully: {run.info.run_id}")
        print(json.dumps(metrics, indent=2))

    print("🚀 All metrics and plots logged to MLflow (DagsHub).")
    return {"model": best, "metrics": metrics}

if __name__ == "__main__":
    from app.data import load_raw_data
    from app.preprocess_reg import preprocess_regresion
    df = load_raw_data()
    X, y, feature_columns = preprocess_regresion(df)
    results = train_and_log_reg(X, y)
    print("Final Metrics:")
    print(json.dumps(results["metrics"], indent=2))
    time.sleep(2)