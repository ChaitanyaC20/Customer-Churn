from app.data import load_raw_data
from app.preprocess import preprocess_customer
from app.train import train_and_log
import joblib
from app.config import FEATURES_FILE

def run_all():
    print("Loading raw data...")
    df = load_raw_data()
    print("Preprocessing...")
    X, y, feature_columns = preprocess_customer(df)

    print(f"Training model on {len(X)} rows / {len(feature_columns)} features ...")
    res = train_and_log(X, y)
    print("Training complete. Metrics:")
    print(res["metrics"])

    # Save feature columns for API usage
    joblib.dump(feature_columns, FEATURES_FILE)
    print("Saved feature columns to", FEATURES_FILE)

if __name__ == "__main__":
    run_all()