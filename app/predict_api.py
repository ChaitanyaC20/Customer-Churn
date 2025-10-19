from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, conint, confloat
from typing import Any, Dict
import pandas as pd
import joblib
from app.config import MODEL_FILE, FEATURES_FILE

router = APIRouter()

class ChurnInput(BaseModel):
    Age: conint(ge=18, le=85) = Field(..., description="Customer age between 18 and 85")
    Married: conint(ge=0, le=1) = Field(..., description="Marital status: 0=No, 1=Yes")
    Number_of_Dependents: conint(ge=0, le=10) = Field(..., description="Number of dependents, 0-10")
    Number_of_Referrals: conint(ge=0, le=10) = Field(..., description="Number of referrals, 0-10")
    Offer: conint(ge=0, le=5) = Field(..., description="Offer type, encoded 0–5")
    Contract: conint(ge=0, le=2) = Field(..., description="Contract type, 0=Month-to-Month, 1=One Year, 2=Two Year")
    Revenue_per_Month: confloat(ge=10, le=200) = Field(..., description="Average monthly revenue, 10–200")
    Engagement_Score: conint(ge=0, le=9) = Field(..., description="Engagement level score, 0–9")

    class Config:
        schema_extra = {
            "example": {
                "Age": 35,
                "Married": 1,
                "Number_of_Dependents": 2,
                "Number_of_Referrals": 3,
                "Offer": 2,
                "Contract": 1,
                "Revenue_per_Month": 85.5,
                "Engagement_Score": 6
            }
        }

@router.post("/predict")
def predict(payload: ChurnInput):
    """Predict churn probability for a given customer input"""
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")

    try:
        feature_cols = joblib.load(FEATURES_FILE)
    except Exception:
        feature_cols = []
        try:
            if "scaler" in model.named_steps:
                feature_cols = list(model.named_steps["scaler"].get_feature_names_out())
        except Exception:
            pass

    data_dict = payload.dict()
    features = {**data_dict.get("extra_features", {}), **{k: v for k, v in data_dict.items() if k != "extra_features"}}

    df = pd.DataFrame([features])

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    df = df[feature_cols] if feature_cols else df

    try:
        probs = model.predict_proba(df)
        preds = model.predict(df)

        probability = float(probs[0].max())
        churn_prediction = int(preds[0])

        if churn_prediction == 1:
            message = f"The customer is likely to churn with a probability of {probability:.2f}."
        else:
            message = f"The customer is not likely to churn with a probability of {probability:.2f}."

        return {
            "message": message,
            "probability": round(probability, 3),
            "churn_prediction": "Likely to Churn" if churn_prediction == 1 else "Likely Not to Churn"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")