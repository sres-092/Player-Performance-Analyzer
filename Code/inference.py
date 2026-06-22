# inference.py
# Loads scaler + trained models and produces predictions for the player_matches.csv
# Produces:
#  - artifacts/new_predictions.csv  (row-level predictions)
#  - artifacts/players_match.csv    (Match ID, Player, True Score, LinearRegression, RandomForest, GradientBoosting)

import os
import pandas as pd
import numpy as np
import joblib

os.makedirs("artifacts", exist_ok=True)

DATA_PATH = "data/player_matches.csv"
ART_DIR = "artifacts"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"Loaded {df.shape[0]} rows from {DATA_PATH}")

eps = 1e-6
df['shot_accuracy'] = df['shots_made'] / (df['shots_attempted'] + eps)
df['points_per_min'] = df['points'] / (df['minutes_played'] + eps)

df = df.sort_values(['player_id', 'date'])
df['last_3_avg_points'] = (
    df.groupby('player_id')['points']
      .rolling(3, min_periods=1)
      .mean()
      .shift(1)
      .reset_index(0, drop=True)
)
df['last_3_avg_points'] = df['last_3_avg_points'].fillna(df['points'].mean())

features = [
    'minutes_played','shot_accuracy','points_per_min','assists','rebounds',
    'turnovers','distance_run_km','fatigue_index','last_3_avg_points'
]

# load scaler and models (expect them in artifacts/)
scaler_path = os.path.join(ART_DIR, "scaler.joblib")
lr_path = os.path.join(ART_DIR, "LinearRegression.joblib")
rf_path = os.path.join(ART_DIR, "RandomForest.joblib")
gb_path = os.path.join(ART_DIR, "GradientBoosting.joblib")

def load_or_error(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"Missing artifact: {path}")

scaler = load_or_error(scaler_path)
models = {
    "LinearRegression": load_or_error(lr_path),
    "RandomForest": load_or_error(rf_path),
    "GradientBoosting": load_or_error(gb_path)
}
print("Loaded scaler and models from artifacts/")

X = df[features].fillna(0)
X_s = scaler.transform(X)

for name, model in models.items():
    df[name + "_pred"] = model.predict(X_s).round(3)

row_out = os.path.join(ART_DIR, "new_predictions.csv")
df.to_csv(row_out, index=False)
print(f"Saved row-level predictions -> {row_out}")

# Map player_id -> name (optional). Edit or leave empty if you prefer ids only.
player_name_map = {
    "P001": "Aarav", "P002": "Laksh", "P003": "Shaurya", "P004": "Kunal",
    "P005": "Aditya", "P006": "Dev", "P007": "Raghav", "P008": "Krish",
    "P009": "Reyansh", "P010": "Yash"
}
def player_display(pid):
    name = player_name_map.get(pid, "")
    return f"{name} ({pid})" if name else pid

out_rows = []
for _, r in df.iterrows():
    match_id = r['match_id']
    player = player_display(r['player_id'])
    if 'performance_score' in df.columns:
        true_score = r['performance_score']
    elif 'coach_rating' in df.columns:
        true_score = r['coach_rating']
    else:
        true_score = ""
    out_rows.append({
        "Match ID": match_id,
        "Player": player,
        "True Score": round(float(true_score), 3) if true_score != "" else "",
        "Linear Regression": float(r["LinearRegression_pred"]),
        "Random Forest": float(r["RandomForest_pred"]),
        "Gradient Boosting": float(r["GradientBoosting_pred"])
    })

players_match_df = pd.DataFrame(out_rows)
players_match_path = os.path.join(ART_DIR, "players_match.csv")
players_match_df.to_csv(players_match_path, index=False)
print(f"Saved aggregated players_match -> {players_match_path}")
