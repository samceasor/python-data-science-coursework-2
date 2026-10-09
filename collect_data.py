"""
PYTHON FOR DATA SCIENCE COURSEWORK 2
Project: Predicting Daily Rainfall Occurrence in Kampala

This script collects the raw data directly from the permitted Open-Meteo
Historical Weather API and saves the response as a CSV file.

Run:
    python collect_data.py

The raw file is saved in data/kampala_weather_raw.csv
"""
import requests
import pandas as pd
from pathlib import Path

API_URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": 0.3476,
    "longitude": 32.5825,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "daily": (
        "temperature_2m_mean,relative_humidity_2m_mean,"
        "precipitation_sum,windspeed_10m_max,pressure_msl_mean"
    ),
    "timezone": "Africa/Kampala",
}

OUT = Path("data/kampala_weather_raw.csv")

def collect_weather():
    response = requests.get(API_URL, params=PARAMS, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "daily" not in payload or "time" not in payload["daily"]:
        raise ValueError("The API response did not contain daily weather data.")

    daily = payload["daily"]
    df = pd.DataFrame(daily)

    required = {
        "time",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "pressure_msl_mean",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Collected {len(df)} daily records.")
    print(f"Saved raw data to: {OUT.resolve()}")

if __name__ == "__main__":
    collect_weather()
