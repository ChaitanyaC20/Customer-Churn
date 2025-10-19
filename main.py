from fastapi import FastAPI
from app.predict_api import router as predict_router
from app.config import API_HOST, API_PORT

app = FastAPI(title="Telco Churn Prediction API")

app.include_router(predict_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API. Use /api/predict to get predictions."}