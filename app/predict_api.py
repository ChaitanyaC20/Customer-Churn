from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import pandas as pd
import joblib
from app.config import MODEL_FILE, FEATURES_FILE

router = APIRouter()

# Define input fields - flexible dict can also be used, but typed model helps docs
class ChurnInput(BaseModel):
    # Add any features you expect; using optional Any allows flexible inputs
    data: Dict[str, Any]

@router.post("/predict")
def predict(payload: ChurnInput):
    # Load model & feature columns
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")

    try:
        feature_cols = joblib.load(FEATURES_FILE)
    except Exception:
        # if features file not available, infer from model pipeline (best effort)
        try:
            feature_cols = list(model.named_steps['scaler'].mean_.shape[0] if 'scaler' in model.named_steps else [])
        except Exception:
            feature_cols = []

    # Build input dataframe
    df = pd.DataFrame([payload.data])
    # Derived features: compute total assets if relevant (example)
    # df['total_assets'] = df.get('residential_assets_value',0) + df.get('commercial_assets_value',0) + ...

    # Ensure all feature columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    df = df[feature_cols] if feature_cols else df

    # Prediction
    try:
        probs = model.predict_proba(df)
        preds = model.predict(df)
        # return predicted label and highest probability
        return {"prediction": int(preds[0]), "probability": float(probs[0].max())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))