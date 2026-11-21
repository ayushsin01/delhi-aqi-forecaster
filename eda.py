"""Quick EDA plots -> assets/. Run: python eda.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from features import load_data

df = load_data("data/delhi-weather-aqi-2025.csv")

df.groupby("location")["pm2_5"].mean().sort_values().plot(kind="barh", figsize=(7, 3.5))
plt.title("Average PM2.5 by station"); plt.xlabel("µg/m³"); plt.tight_layout()
plt.savefig("assets/eda_station.png", dpi=130); plt.close()

df.groupby(df["datetime"].dt.month)["pm2_5"].mean().plot(marker="o", figsize=(7, 3.5))
plt.title("Monthly average PM2.5"); plt.xlabel("Month"); plt.tight_layout()
plt.savefig("assets/eda_monthly.png", dpi=130); plt.close()

df.groupby(df["datetime"].dt.hour)["pm2_5"].mean().plot(marker="o", figsize=(7, 3.5))
plt.title("PM2.5 by hour of day"); plt.xlabel("Hour"); plt.tight_layout()
plt.savefig("assets/eda_hourly.png", dpi=130); plt.close()

cols = ["temp_c", "humidity", "pressure_mb", "windspeed_kph", "pm2_5", "pm10", "co", "no2"]
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap"); plt.tight_layout()
plt.savefig("assets/eda_corr.png", dpi=130); plt.close()
print("Saved EDA plots to assets/")
