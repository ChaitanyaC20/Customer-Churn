from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, conint, confloat
from typing import Any, Dict
import pandas as pd
import joblib
import numpy as np
from app.config import MODEL_FILE_REG, FEATURES_FILE_REG

router = APIRouter()

class RevenueInput(BaseModel):
    Age: conint(ge=18, le=85) = Field(..., description="Customer age between 18 and 85")
    Married: conint(ge=0, le=1) = Field(..., description="Marital status: 0=No, 1=Yes")
    Number_of_Dependents: conint(ge=0, le=10) = Field(..., description="Number of dependents, 0-10")
    Number_of_Referrals: conint(ge=0, le=10) = Field(..., description="Number of referrals, 0-10")
    Offer: conint(ge=0, le=5) = Field(..., description="Offer type, encoded 0–5")
    Contract: conint(ge=0, le=2) = Field(..., description="Contract type, 0=Month-to-Month, 1=One Year, 2=Two Year")
    Tenure_in_Months: confloat(ge=1, le=72) = Field(..., description="Average monthly revenue, 1–72")
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
                "Tenure_in_Months": 10,
                "Engagement_Score": 6
            }
        }

@router.post("/predict_revenue")
def predict(payload: RevenueInput):
    """Predict total revenue for a given customer input"""
    try:
        model = joblib.load(MODEL_FILE_REG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")

    try:
        feature_cols = joblib.load(FEATURES_FILE_REG)
    except Exception:
        feature_cols = []
        
    data = pd.DataFrame([payload.dict()])

    for c in feature_cols:
        if c not in data.columns:
            data[c] = 0

    if feature_cols:
        data = data[feature_cols]

    try:
        prediction = model.predict(data)
        predicted_revenue = float(np.round(prediction[0], 2))

        message = (
            f"The predicted total revenue for this customer is approximately {predicted_revenue:,.2f}."
        )

        return {
            "predicted_revenue": predicted_revenue,
            "message": message,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")