This project focuses on analyzing the Telco Customer Churn dataset to predict customer churn (classification) and customer's total revenue (regression).  
The project demonstrates an end-to-end MLOps pipeline — from data preprocessing and model training to api and tracking with MLflow and Dagshub.

Project Overview :

The Telco Customer Churn Dataset contains 5 input files with customer demographic, status and service-related data.  
The goal is twofold:
1. Classification — Predict whether a customer is likely to churn (Yes / No) using XGBClassifier.
2. Regression — Predict a customer's total revenue using LGBMRegressor.

FastAPI-based API for real-time interaction and easy integration.

MLflow + DagsHub Integration for experiment tracking.


Features Used : 

Age, Married, Number_of_Dependents, Number_of_Referrals, Offer, Contract, Engagement_Score, Revenue_per_Month(classification only), Tenure_in_Months(Regression only)

Installation :

Clone the Repository : git clone https://github.com/ChaitanyaC20/Customer-churn.git cd Customer-churn.

Create and Activate Environment : python -m venv py310env py310env\Scripts\activate

Install Requirements : pip install -r requirements.txt

Load the data : python -m app.data

Preprocess the data : python -m app.preprocess

Training the both the  Model : python -m app.train & python -m app.train_reg

Run full pipeline. run_pipeline.py & run_pipeline_reg.py

Running the API : uvicorn main:app --reload

Results :

<img width="1377" height="740" alt="image" src="https://github.com/user-attachments/assets/0c14e0de-9add-4850-9561-08df3f5be845" />

<img width="1371" height="710" alt="image" src="https://github.com/user-attachments/assets/9af06c90-7fcc-4be4-a8ca-0f26e5d8c691" />


Experiment Tracking :

All MLflow experiment runs, metrics, and model artifacts for this project are tracked on DagsHub.

Classification : [https://dagshub.com/ChaitanyaC20/Customer_Churn.mlflow/#/experiments](https://dagshub.com/ChaitanyaC20/Customer_Churn.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)
Regression : [https://dagshub.com/ChaitanyaC20/Total_Revenue_Regression.mlflow/#/experiments](https://dagshub.com/ChaitanyaC20/Total_Revenue_Regression.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)


