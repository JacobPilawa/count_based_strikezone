import pandas as pd
from pybaseball import statcast
import os

DATA_DIR = "../data"

YEAR_RANGES = {
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-28", "2019-09-30"),
    2020: ("2020-07-23", "2020-09-27"),  # Shortened COVID season
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-23", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-20", "2025-09-28"),
}

def download_year(year):
    path = os.path.join(DATA_DIR, f"statcast_{year}.parquet")
    if os.path.exists(path):
        print(f"Loading cached data for {year}...")
        return pd.read_parquet(path)
    
    start_dt, end_dt = YEAR_RANGES.get(year, (f"{year}-03-01", f"{year}-10-31"))
    print(f"Downloading {year} MLB Statcast data...")
    data = statcast(start_dt=start_dt, end_dt=end_dt, verbose=True)
    
    print(f"Downloaded {len(data)} pitches for {year}")
    data.to_parquet(path)
    print(f"Cached to {path}")
    return data

if __name__ == "__main__":
    for year in range(2018, 2026):
        download_year(year)
    print("\nAll data downloaded!")
