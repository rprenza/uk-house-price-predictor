"""
train_lightweight_model.py
──────────────────────────
Run this script ONCE to generate the lightweight model files
needed for Streamlit Cloud deployment.

Output files (all small enough for GitHub):
  - model_xgboost.pkl       (~1-2 MB)
  - features_list.pkl       (< 1 KB)
  - geo_medians.pkl         (< 1 MB)

Usage:
  python train_lightweight_model.py
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 50)
print("  Lightweight Model Training for Deployment")
print("=" * 50)

# ── 1. Load dataset ───────────────────────────────
print("\n[1/5] Loading dataset_ml.csv ...")
df = pd.read_csv("dataset_ml.csv")
print(f"      Full dataset: {len(df):,} rows")

# ── 2. Sample 100k rows ───────────────────────────
SAMPLE_SIZE = 100_000
if len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42)
    print(f"      Sampled:      {len(df):,} rows (stratified for speed)")
else:
    print(f"      Using full dataset: {len(df):,} rows")

# ── 3. Features & target ──────────────────────────
print("\n[2/5] Preparing features ...")
target   = "price"
features = [c for c in df.columns if c != target]
X = df[features]
y = df[target]
print(f"      Features: {features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. Train XGBoost ──────────────────────────────
print("\n[3/5] Training XGBoost ...")
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
print(f"      RMSE: £{rmse:,.0f}")
print(f"      R²:   {r2:.4f}")

# ── 5. Save model & features ──────────────────────
print("\n[4/5] Saving model files ...")
with open("model_xgboost.pkl", "wb") as f:
    pickle.dump(model, f)
print("      Saved: model_xgboost.pkl")

with open("features_list.pkl", "wb") as f:
    pickle.dump(features, f)
print("      Saved: features_list.pkl")

# ── 6. Save geo medians ───────────────────────────
print("\n[5/5] Saving geographic medians ...")

# Re-load full dataset for accurate medians
df_full = pd.read_csv("dataset_ml.csv")

geo = {
    "county_median": df_full.groupby("county")["price"].median().to_dict()
                     if "county" in df_full.columns else {},
    "town_median":   df_full.groupby("town")["price"].median().to_dict()
                     if "town"   in df_full.columns else {},
    "town_county":   df_full[["county","town"]].dropna()
                     .drop_duplicates().to_dict(orient="records")
                     if "county" in df_full.columns and "town" in df_full.columns
                     else [],
}

with open("geo_medians.pkl", "wb") as f:
    pickle.dump(geo, f)
print("      Saved: geo_medians.pkl")

# ── Summary ───────────────────────────────────────
import os
print("\n" + "=" * 50)
print("  Done! Files generated:")
for fname in ["model_xgboost.pkl", "features_list.pkl", "geo_medians.pkl"]:
    size_kb = os.path.getsize(fname) / 1024
    print(f"    {fname:<25} {size_kb:,.0f} KB")
print("\n  Next step: update app.py and push to GitHub.")
print("=" * 50)
