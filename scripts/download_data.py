import pandas as pd
from pybaseball import statcast
import os

DATA_PATH = "../data/statcast_2025.parquet"

def download_2025_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading cached data from {DATA_PATH}")
        return pd.read_parquet(DATA_PATH)
    
    print("Downloading 2025 MLB Statcast data...")
    print("This may take a while for the full season...")
    
    data = statcast(start_dt="2025-03-20", end_dt="2025-09-28", verbose=True)
    
    print(f"Downloaded {len(data)} pitches")
    
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    data.to_parquet(DATA_PATH)
    print(f"Cached data to {DATA_PATH}")
    
    return data

if __name__ == "__main__":
    df = download_2025_data()
    print(f"Columns available: {df.columns.tolist()}")
    print(f"Total pitches: {len(df)}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
