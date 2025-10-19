import os

DATA_DIR = os.getenv("DATA_DIR", r"D:\Data science\Churn Project")

DEMOGRAPHICS_PATH = os.getenv("DEMOGRAPHICS_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_demographics.xlsx"))
LOCATION_PATH = os.getenv("LOCATION_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_location.xlsx"))
POPULATION_PATH = os.getenv("POPULATION_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_population.xlsx"))
SERVICES_PATH = os.getenv("SERVICES_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_services.xlsx"))
STATUS_PATH = os.getenv("STATUS_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_status.xlsx"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/ChaitanyaC20/Customer_Churn.mlflow")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "churn_experiment")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "Churn_XGB_Model")

# Model/artifact storage (local)
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "runs")
MODEL_FILE = os.path.join(ARTIFACT_DIR, "model.pkl")
FEATURES_FILE = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")
METRICS_FILE = os.path.join(ARTIFACT_DIR, "metrics.json")

# FastAPI settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))