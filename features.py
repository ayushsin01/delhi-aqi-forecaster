"""Shared data loading + feature engineering (used by train.py and app.py)."""
import numpy as np
import pandas as pd

HORIZON = 6  # predict PM2.5 this many hours ahead
WEATHER = ["temp_c", "humidity", "pressure_mb", "windspeed_kph"]
POLLUTANTS = ["pm2_5", "pm10", "co", "no2"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(
        df["date_ist"] + " " + df["time_ist"], format="%d/%m/%Y %H:%M"
    )
    df = df.sort_values(["location", "datetime"]).reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """Build lag/rolling/time features per station (no leakage across stations)."""
    out = []
    for loc, g in df.groupby("location", sort=False):
        g = g.sort_values("datetime").copy()
        for lag in (1, 3, 6, 12, 24):
            g[f"pm25_lag{lag}"] = g["pm2_5"].shift(lag)
        g["pm25_roll6"] = g["pm2_5"].rolling(6).mean()
        g["pm25_roll24"] = g["pm2_5"].rolling(24).mean()
        g["wind_roll6"] = g["windspeed_kph"].rolling(6).mean()
        g["hour"] = g["datetime"].dt.hour
        g["month"] = g["datetime"].dt.month
        g["dayofweek"] = g["datetime"].dt.dayofweek
        # cyclical encoding so 23:00 and 00:00 are "close"
        g["hour_sin"] = np.sin(2 * np.pi * g["hour"] / 24)
        g["hour_cos"] = np.cos(2 * np.pi * g["hour"] / 24)
        if with_target:
            g["target"] = g["pm2_5"].shift(-HORIZON)
        out.append(g)
    feat = pd.concat(out).reset_index(drop=True)
    feat = pd.get_dummies(feat, columns=["location"], prefix="loc", dtype=int)
    return feat


def feature_columns(feat: pd.DataFrame) -> list:
    base = (
        POLLUTANTS + WEATHER
        + [c for c in feat.columns if c.startswith(("pm25_", "wind_roll"))]
        + ["hour_sin", "hour_cos", "dayofweek"]  # no "month": only 1 year of data, cannot generalise across seasons
        + [c for c in feat.columns if c.startswith("loc_")]
    )
    return base


def pm25_to_category(pm25: float) -> str:
    """CPCB (India) PM2.5 24h breakpoints."""
    if pm25 <= 30:
        return "Good"
    if pm25 <= 60:
        return "Satisfactory"
    if pm25 <= 90:
        return "Moderate"
    if pm25 <= 120:
        return "Poor"
    if pm25 <= 250:
        return "Very Poor"
    return "Severe"
