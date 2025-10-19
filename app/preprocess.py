# app/preprocess.py
import numpy as np
import pandas as pd

def preprocess_customer(df: pd.DataFrame):
    """Apply mappings, feature engineering and return X, y, feature_columns and label encoder."""
    df = df.copy()

    # Fill missing
    df["Offer"] = df["Offer"].fillna("Not Avail")
    df["Internet_Type"] = df["Internet_Type"].fillna("Not opt")

    # Mappings
    df["Offer"] = df["Offer"].map({"Not Avail":0,"Offer A":1,"Offer B":2,"Offer C":3,"Offer D":4,"Offer E":5}).astype("int32")
    df["Internet_Type"] = df["Internet_Type"].map({"Not opt":0,"Cable":1,"DSL":2,"Fiber Optic":3}).astype("int32")
    df["Payment_Method"] = df["Payment_Method"].map({"Mailed Check":0,"Credit Card":1,"Bank Withdrawal":2}).astype("int32")
    df["Contract"] = df["Contract"].map({"Month-to-Month":0,"One Year":1,"Two Year":2}).astype("int32")
    df["Gender"] = df["Gender"].map({"Male":0,"Female":1}).astype("int32")

    # Binary yes/no columns -> 0/1
    binary_cols = [
        "Under_30","Senior_Citizen","Married","Dependents","Referred_a_Friend",
        "Multiple_Lines","Internet_Service","Phone_Service","Online_Security",
        "Online_Backup","Device_Protection_Plan","Premium_Tech_Support",
        "Streaming_TV","Streaming_Movies","Streaming_Music","Unlimited_Data","Paperless_Billing"
    ]
    for c in binary_cols:
        if c in df.columns:
            df[c] = df[c].map({"No":0,"Yes":1}).astype("int32")

    df['Revenue_per_Month'] = df['Total_Revenue'] / df['Tenure_in_Months'].replace(0, np.nan)

    df["Engagement_Score"] = df["Streaming_TV"] + df["Streaming_Movies"]+ df["Streaming_Music"] + df["Online_Security"] + df["Device_Protection_Plan"] + df["Premium_Tech_Support"] + df["Online_Backup"] + df["Unlimited_Data"] + df["Internet_Service"]

    # Drop list (from your original)
    columns_to_exclude = [
    "Customer_ID", "Lat_Long", "Country", "Quarter_x", "Generated_Reviews", "CLTV", "Streaming_TV", "Internet_Service",
    "Quarter_y", "ID", "Satisfaction_Score", "Customer_Status", "Churn_Label", "Churn_Score", "Streaming_Music",
    "Senior_Citizen", "Multiple_Lines", "Log_Total_Revenue", "Churn_Category", "Churn_Reason", "Streaming_Movies", 
    "State", "Zip_Code", "City", "Total_Long_Distance_Charges", "Paperless_Billing", "Online_Security", 
    "Total_Extra_Data_Charges", "Total_Refunds", "Total_Charges", "Monthly_Charge",
    "Dependents", "Referred_a_Friend", "Under_30", "Internet_Type", "Online_Backup", "Device_Protection_Plan",
    "Payment_Method", "Population", "Latitude", "Longitude", "Total_Revenue", "Premium_Tech_Support",
    "Tenure_in_Months", "Gender", "Phone_Service", "Unlimited_Data", "Avg_Monthly_Long_Distance_Charges"
]
    df = df.drop(columns=[c for c in columns_to_exclude if c in df.columns], errors='ignore')

    X = df.drop(columns=["Churn_Value"], errors='ignore')
    y = df["Churn_Value"]

    # Replace NaNs (simple approach)
    X = X.fillna(0)

    feature_columns = list(X.columns)
    return X, y, feature_columns

if __name__ == "__main__":
    from app.data import load_raw_data  

    try:
        df = load_raw_data()
        print("Raw data loaded successfully!")
        print(f"Shape: {df.shape}")

        X, y, feature_columns = preprocess_customer(df)

        print("\nPreprocessing complete!")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(X.columns)
        print(f"Number of features: {len(feature_columns)}")
        print(y.value_counts())
    except Exception as e:
        import traceback
        print("Error during preprocessing:")
        traceback.print_exc()