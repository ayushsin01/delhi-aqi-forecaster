"""Streamlit app. Run:  streamlit run app.py"""
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from features import load_data, make_features, pm25_to_category

st.set_page_config(page_title="Delhi Air Quality Forecaster", page_icon="🌫️", layout="wide")

COLORS = {"Good": "#2e7d32", "Satisfactory": "#7cb342", "Moderate": "#f9a825",
          "Poor": "#ef6c00", "Very Poor": "#d32f2f", "Severe": "#6a1b1a"}


@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")


@st.cache_data
def load_everything():
    raw = load_data("data/delhi-weather-aqi-2025.csv")
    feat = make_features(raw, with_target=False)
    return raw, feat


bundle = load_model()
model, cols, H = bundle["model"], bundle["columns"], bundle["horizon"]
raw, feat = load_everything()
metrics = json.load(open("models/metrics.json"))

st.title("🌫️ Delhi Air Quality Forecaster")
st.caption(f"Predicts PM2.5 **{H} hours ahead** for 6 Delhi monitoring stations "
           f"using weather + recent pollution history ({metrics['best_model']}).")

# ---------------- sidebar ----------------
stations = sorted(raw["location"].unique())
station = st.sidebar.selectbox("Station", stations)
min_d, max_d = raw["datetime"].min().date(), raw["datetime"].max().date()
day = st.sidebar.date_input("Date", value=pd.Timestamp("2025-12-10").date(),
                            min_value=min_d + pd.Timedelta(days=2),
                            max_value=max_d - pd.Timedelta(days=1))
hour = st.sidebar.slider("Hour of day", 0, 23, 18)
when = pd.Timestamp(day) + pd.Timedelta(hours=hour)

st.sidebar.markdown("---")
r = metrics["results"]
st.sidebar.markdown("**Test performance (Oct–Dec 2025)**")
st.sidebar.metric("Model MAE", f"{r[metrics['best_model']]['MAE']} µg/m³")
st.sidebar.metric("Baseline MAE", f"{r['Baseline (persistence)']['MAE']} µg/m³")

# ---------------- prediction ----------------
loc_col = f"loc_{station}"
row = feat[(feat["datetime"] == when) & (feat.get(loc_col, 0) == 1)]
row = row.dropna(subset=cols)

if row.empty:
    st.warning("Not enough history for this time. Pick a later date.")
    st.stop()

current = float(row["pm2_5"].iloc[0])
pred = current + float(model.predict(row[cols])[0])
pred = max(pred, 0.0)
cat_now, cat_pred = pm25_to_category(current), pm25_to_category(pred)

target_time = when + pd.Timedelta(hours=H)
actual_row = raw[(raw["location"] == station) & (raw["datetime"] == target_time)]
actual = float(actual_row["pm2_5"].iloc[0]) if not actual_row.empty else None

c1, c2, c3 = st.columns(3)
c1.metric(f"Now ({when:%d %b, %H:%M})", f"{current:.0f} µg/m³")
c1.markdown(f"<span style='color:{COLORS[cat_now]};font-weight:600'>{cat_now}</span>",
            unsafe_allow_html=True)
c2.metric(f"Predicted +{H}h ({target_time:%H:%M})", f"{pred:.0f} µg/m³",
          delta=f"{pred - current:+.0f}", delta_color="inverse")
c2.markdown(f"<span style='color:{COLORS[cat_pred]};font-weight:600'>{cat_pred}</span>",
            unsafe_allow_html=True)
if actual is not None:
    c3.metric("Actual (from data)", f"{actual:.0f} µg/m³", delta=f"error {pred - actual:+.0f}",
              delta_color="off")

# ---------------- chart ----------------
hist = raw[(raw["location"] == station) &
           (raw["datetime"] >= when - pd.Timedelta(hours=48)) &
           (raw["datetime"] <= target_time + pd.Timedelta(hours=6))]
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(hist["datetime"], hist["pm2_5"], label="Actual PM2.5", color="#1f77b4")
ax.axvline(when, color="grey", ls="--", lw=1)
ax.scatter([target_time], [pred], color="red", zorder=5, s=60, label=f"Predicted (+{H}h)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend(); fig.autofmt_xdate(); plt.tight_layout()
st.pyplot(fig)

with st.expander("Weather at selected time"):
    w = row[["temp_c", "humidity", "pressure_mb", "windspeed_kph"]].iloc[0]
    st.write(w.rename({"temp_c": "Temp (°C)", "humidity": "Humidity (%)",
                       "pressure_mb": "Pressure (mb)", "windspeed_kph": "Wind (kph)"}))

with st.expander("About this project"):
    st.markdown(
        f"- Data: hourly weather + air quality, 6 Delhi stations, 2025.\n"
        f"- Features: current pollutants, weather, PM2.5 lags/rolling means, time of day.\n"
        f"- Model learns the *change* in PM2.5 over {H}h; test split is time-based "
        f"(train Jan–Sep, test Oct–Dec).\n"
        f"- Categories follow CPCB PM2.5 breakpoints."
    )
