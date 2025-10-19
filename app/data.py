# app/data.py
import pandas as pd
from app.config import DEMOGRAPHICS_PATH, LOCATION_PATH, POPULATION_PATH, SERVICES_PATH, STATUS_PATH

def load_raw_data():
    """Load and merge the Excel files into a single DataFrame."""
    dem = pd.read_excel(DEMOGRAPHICS_PATH)
    loc = pd.read_excel(LOCATION_PATH)
    pop = pd.read_excel(POPULATION_PATH)
    srv = pd.read_excel(SERVICES_PATH)
    st = pd.read_excel(STATUS_PATH)

    c1 = pd.merge(dem, loc, on="Customer ID")
    c2 = pd.merge(srv, st, on="Customer ID")
    df = pd.merge(c1, c2, on="Customer ID")
    df = pd.merge(df, pop, how="left", on="Zip Code")
    df.columns = df.columns.str.replace(" ", "_")
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")