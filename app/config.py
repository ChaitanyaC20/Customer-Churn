import os

DATA_DIR = os.getenv("DATA_DIR", r"D:\Data science\Churn Project")

DEMOGRAPHICS_PATH = os.getenv("DEMOGRAPHICS_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_demographics.xlsx"))
LOCATION_PATH = os.getenv("LOCATION_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_location.xlsx"))
POPULATION_PATH = os.getenv("POPULATION_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_population.xlsx"))
SERVICES_PATH = os.getenv("SERVICES_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_services.xlsx"))
STATUS_PATH = os.getenv("STATUS_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_status.xlsx"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/ChaitanyaC20/Customer_Churn.mlflow")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "churn_experiment")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "Churn_XGB_Model")

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "runs")
MODEL_FILE = os.path.join(ARTIFACT_DIR, "model.pkl")
FEATURES_FILE = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")
METRICS_FILE = os.path.join(ARTIFACT_DIR, "metrics.json")

ARTIFACT_DIR_REG   = os.getenv("ARTIFACT_DIR_REG", "runs_revenue")
MODEL_FILE_REG   = os.path.join(ARTIFACT_DIR_REG, "model.pkl")
FEATURES_FILE_REG   = os.path.join(ARTIFACT_DIR_REG, "feature_columns.pkl")
METRICS_FILE_REG   = os.path.join(ARTIFACT_DIR_REG, "metrics.json")

MLFLOW_TRACKING_URI_REG = os.getenv("MLFLOW_TRACKING_URI_REG", "https://dagshub.com/ChaitanyaC20/Total_Revenue_Regression.mlflow")
MLFLOW_EXPERIMENT_Reg = os.getenv("MLFLOW_EXPERIMENT_Reg", "Regression")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "Reg_LGBM_Model")