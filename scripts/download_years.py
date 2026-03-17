import pandas as pd
from pybaseball import statcast
import os

DATA_DIR = "../data"

def download_year(year):
    path = os.path.join(DATA_DIR, f"statcast_{year}.parquet")
    if os.path.exists(path):
        print(f"Loading cached data for {year}...")
        return pd.read_parquet(path)
    
    print(f"Downloading {year} MLB Statcast data...")
    if year == 2025:
        data = statcast(start_dt="2025-03-20", end_dt="2025-09-28", verbose=True)
    elif year == 2024:
        data = statcast(start_dt="2024-03-20", end_dt="2024-09-29", verbose=True)
    else:
        data = statcast(start_dt="2023-03-23", end_dt="2023-10-01", verbose=True)
    
    print(f"Downloaded {len(data)} pitches for {year}")
    data.to_parquet(path)
    print(f"Cached to {path}")
    return data

if __name__ == "__main__":
    for year in [2023, 2024, 2025]:
        download_year(year)
    print("\nAll data downloaded!")
