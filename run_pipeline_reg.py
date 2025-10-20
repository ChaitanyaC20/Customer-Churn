from app.data import load_raw_data
from app.preprocess_reg import preprocess_regresion
from app.train_reg import train_and_log_reg
import joblib
from app.config import FEATURES_FILE_REG

def run_all():
    print("Loading raw data...")
    df = load_raw_data()
    print("Preprocessing...")
    X, y, feature_columns = preprocess_regresion(df)

    print(f"Training model on {len(X)} rows / {len(feature_columns)} features ...")
    res = train_and_log_reg(X, y)
    print("Training complete. Metrics:")
    print(res["metrics"])

    joblib.dump(feature_columns, FEATURES_FILE_REG)
    print("Saved feature columns to", FEATURES_FILE_REG)

if __name__ == "__main__":
    run_all()