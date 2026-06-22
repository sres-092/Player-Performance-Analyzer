# generate_synthetic_data.py
# This file generates sample player performance data for the ML model.

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ensure data directory
os.makedirs("data", exist_ok=True)

n_players = 30
n_matches = 150

players = [f"P{str(i).zfill(3)}" for i in range(1, n_players+1)]
dates = [datetime(2024, 1, 1) + timedelta(days=int(x)) for x in np.random.choice(range(200), n_matches)]

rows = []
for i in range(n_matches):
    match_id = f"M{1000+i}"
    date = dates[i].strftime("%Y-%m-%d")
    # pick six players per match
    chosen = list(np.random.choice(players, size=6, replace=False))
    for p in chosen:
        minutes_played = np.random.randint(15, 90)
        shots_attempted = np.random.randint(5, 25)
        # shots_made cannot exceed attempts
        prob_make = 0.35 + np.random.rand()*0.3
        shots_made = int(np.random.binomial(shots_attempted, prob_make))
        points = shots_made + np.random.randint(0, 5)
        assists = int(np.random.poisson(2))
        rebounds = int(np.random.poisson(3))
        turnovers = int(np.random.poisson(1))
        distance_run_km = round(np.random.uniform(1, 8), 2)
        fatigue_index = round(np.random.uniform(0, 1), 2)
        # synthetic coach_rating target (clipped 1-10)
        coach_rating = round(
            float(np.clip(
                4 + 0.08*points + 0.35*shots_made + 0.12*assists + 0.05*rebounds
                - 0.4*turnovers - 2.5*fatigue_index + np.random.normal(0,1.2),
                1, 10
            )), 2
        )
        rows.append([
            match_id, date, "TeamA", "TeamB", p, minutes_played, shots_attempted,
            shots_made, points, assists, rebounds, turnovers, distance_run_km,
            fatigue_index, coach_rating
        ])

df = pd.DataFrame(rows, columns=[
    "match_id", "date", "team", "opponent", "player_id",
    "minutes_played", "shots_attempted", "shots_made", "points",
    "assists", "rebounds", "turnovers", "distance_run_km",
    "fatigue_index", "coach_rating"
])

out_path = "data/player_matches.csv"
df.to_csv(out_path, index=False)
print(f"✅ Data generated successfully and saved in '{out_path}' (rows={len(df)})")
