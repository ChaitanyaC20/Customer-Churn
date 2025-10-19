# test_dagshub_run.py
import dagshub
dagshub.init(repo_owner='ChaitanyaC20', repo_name='Customer_Churn', mlflow=True)

import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("accuracy", 0.89)

print("✅ Logged test run successfully to DagsHub!")