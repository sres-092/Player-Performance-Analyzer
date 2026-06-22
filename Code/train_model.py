# train_model.py
# Trains Linear Regression, Random Forest and Gradient Boosting models


import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def evaluate_and_save(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def plot_feature_importance(features, importances, out_path, title="Feature importance"):
    idx = np.argsort(importances)
    plt.figure(figsize=(8,6))
    plt.barh(range(len(idx)), importances[idx], align='center')
    plt.yticks(range(len(idx)), [features[i] for i in idx])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_linear_coefficients(features, coefs, out_path):
    idx = np.argsort(np.abs(coefs))
    plt.figure(figsize=(8,6))
    plt.barh(range(len(idx)), coefs[idx], align='center')
    plt.yticks(range(len(idx)), [features[i] for i in idx])
    plt.xlabel('Coefficient value')
    plt.title('Linear Regression coefficients (sorted by absolute value)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    try:
        print("📂 Loading dataset...")
        df = pd.read_csv("data/player_matches.csv", parse_dates=["date"])
        print(f"Dataset shape: {df.shape}")
        # Feature engineering
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
        target = 'coach_rating'

        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            raise ValueError(f"Missing expected feature columns: {missing_feats}")

        df = df.dropna(subset=[target])
        X = df[features].fillna(0)
        y = df[target].astype(float)

        print(f"Using {len(features)} features. Rows: {X.shape[0]}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ART_DIR = "artifacts"
        ensure_dir(ART_DIR)

        joblib.dump(scaler, os.path.join(ART_DIR, "scaler.joblib"))

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
        }

        metrics_rows = []
        preds_out = X_test.copy()
        preds_out['true_score'] = y_test.values
        preds_out.index.name = "idx"

        for name, model in models.items():
            print(f"\n🔧 Training {name} ...")
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            metrics = evaluate_and_save(y_test, y_pred)
            metrics_row = {"Model": name, **metrics}
            metrics_rows.append(metrics_row)
            print(f"{name} metrics: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}")

            model_path = os.path.join(ART_DIR, f"{name}.joblib")
            joblib.dump(model, model_path)
            print(f"💾 Saved model -> {model_path}")

            preds_out[name + "_pred"] = y_pred

            if name in ("RandomForest", "GradientBoosting"):
                try:
                    importances = model.feature_importances_
                    out_png = os.path.join(ART_DIR, f"feature_importance_{name.lower()}.png")
                    plot_feature_importance(features, importances, out_png,
                                             title=f"Feature importance — {name}")
                    print(f"📊 Feature importance saved -> {out_png}")
                except Exception as e:
                    print(f"⚠️ Could not plot feature importance for {name}: {e}")
            elif name == "LinearRegression":
                try:
                    coefs = model.coef_
                    out_png = os.path.join(ART_DIR, "linear_coefficients.png")
                    plot_linear_coefficients(features, coefs, out_png)
                    print(f"📈 Linear regression coefficients saved -> {out_png}")
                except Exception as e:
                    print(f"⚠️ Could not plot linear coefficients: {e}")

        metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
        metrics_df.to_csv(os.path.join(ART_DIR, "model_metrics.csv"), index=False)
        preds_out.to_csv(os.path.join(ART_DIR, "predictions_test_set.csv"), index=True)

        print("\n✅ All models trained and artifacts saved in 'artifacts/'")
        print(metrics_df)

    except Exception as exc:
        print("❌ Error during training:", str(exc))
        raise

if __name__ == "__main__":
    main()
