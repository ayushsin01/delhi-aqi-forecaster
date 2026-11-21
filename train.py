"""Train + evaluate models. Run:  python train.py

Target = PM2.5 `HORIZON` hours ahead. Models learn the *change* from current PM2.5
(delta), which handles the winter-vs-summer distribution shift much better than
predicting the raw value.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from features import HORIZON, feature_columns, load_data, make_features

DATA = "data/delhi-weather-aqi-2025.csv"
SPLIT_DATE = "2025-10-01"  # train: before, test: from this date (time-based split!)


def metrics(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def main():
    df = load_data(DATA)
    feat = make_features(df).dropna().reset_index(drop=True)
    cols = feature_columns(feat)
    feat["delta"] = feat["target"] - feat["pm2_5"]

    train = feat[feat["datetime"] < SPLIT_DATE]
    test = feat[feat["datetime"] >= SPLIT_DATE]
    X_tr, d_tr = train[cols], train["delta"]
    X_te, y_te = test[cols], test["target"]
    print(f"Train rows: {len(train)} | Test rows: {len(test)} | Horizon: {HORIZON}h")

    results = {"Baseline (persistence)": metrics(y_te, test["pm2_5"])}
    models = {
        "Ridge": Ridge(alpha=10),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=10, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        ),
    }
    preds = {}
    for name, m in models.items():
        m.fit(X_tr, d_tr)
        preds[name] = test["pm2_5"].values + m.predict(X_te)   # current + predicted change
        results[name] = metrics(y_te, preds[name])

    for k, v in results.items():
        print(f"{k:25s} {v}")

    best_name = min(models, key=lambda n: results[n]["MAE"])
    best = models[best_name]
    print("Best model:", best_name)

    # Refit best on ALL data for the deployed model
    best.fit(feat[cols], feat["delta"])
    joblib.dump({"model": best, "columns": cols, "horizon": HORIZON, "predicts_delta": True},
                "models/model.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "horizon_hours": HORIZON,
                   "split_date": SPLIT_DATE, "results": results}, f, indent=2)

    # ---- plots for README ----
    t = test.copy()
    t["pred"] = preds[best_name]
    loc_col = [c for c in cols if c.startswith("loc_")][0]
    s = t[t[loc_col] == 1].tail(24 * 7)
    plt.figure(figsize=(10, 4))
    plt.plot(s["datetime"], s["target"], label="Actual")
    plt.plot(s["datetime"], s["pred"], label=f"Predicted ({best_name})")
    plt.title(f"{HORIZON}h-ahead PM2.5: actual vs predicted ({loc_col.replace('loc_', '')})")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend(); plt.tight_layout()
    plt.savefig("assets/actual_vs_pred.png", dpi=130); plt.close()

    names = list(results.keys())
    plt.figure(figsize=(7, 3.5))
    plt.barh(names, [results[n]["MAE"] for n in names])
    plt.xlabel("MAE in µg/m³ (lower is better)")
    plt.title("Model comparison on held-out test set (Oct-Dec)")
    plt.tight_layout(); plt.savefig("assets/model_comparison.png", dpi=130); plt.close()

    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=cols).sort_values().tail(12)
        plt.figure(figsize=(7, 4.5)); imp.plot(kind="barh")
        plt.title("Top feature importances")
        plt.tight_layout(); plt.savefig("assets/feature_importance.png", dpi=130); plt.close()

    print("Saved model + metrics + plots.")


if __name__ == "__main__":
    main()
