from fastapi import FastAPI
from app.predict_api import router as churn_router
from app.predict_reg_api import router as revenue_router
from app.config import API_HOST, API_PORT

app = FastAPI(
    title="Telco Analytics API",
    description="Unified API for Customer Churn Classification and Total Revenue Regression",
    version="2.0.0"
)

app.include_router(churn_router, prefix="/api/churn", tags=["Customer Churn Prediction"])
app.include_router(revenue_router, prefix="/api/revenue", tags=["Revenue Prediction"])

@app.get("/")
def root():
    return {
        "message": "Welcome to Telco Analytics API.",
        "available_endpoints": {
            "churn_prediction": "/api/churn/predict",
            "revenue_prediction": "/api/revenue/predict_revenue"
        }
    }