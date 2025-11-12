import os, json
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

@dataclass
class Config:
    data_csv: str = "data/sample_owid_energy_co2.csv"
    outputs_dir: str = "outputs"
    test_start_year: int = 2018
    target_col: str = "co2_mt"
    feature_cols: Tuple[str, ...] = (
        "coal_consumption_twh",
        "oil_consumption_twh",
        "gas_consumption_twh",
        "renewables_consumption_twh",
        "gdp_per_capita_usd",
        "population",
    )

def maybe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["country", "year"])
    df["year"] = df["year"].astype(int)
    return df

def train_test_split_by_year(df: pd.DataFrame, test_start_year: int, target_col: str, feature_cols: Tuple[str, ...]):
    train_df = df[df["year"] < test_start_year].copy()
    test_df = df[df["year"] >= test_start_year].copy()
    X_train = train_df[list(feature_cols)].values
    y_train = train_df[target_col].values
    X_test = test_df[list(feature_cols)].values
    y_test = test_df[target_col].values
    return (X_train, y_train), (X_test, y_test), train_df, test_df

def metrics(y_true, y_pred) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def run_all(config: Config) -> Dict[str, Dict[str, float]]:
    maybe_makedirs(config.outputs_dir)
    df = load_data(config.data_csv)
    (X_train, y_train), (X_test, y_test), train_df, test_df = train_test_split_by_year(
        df, config.test_start_year, config.target_col, config.feature_cols
    )
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.001, random_state=42, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    results, preds_hold = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = metrics(y_test, y_pred)
        preds_hold[name] = y_pred
    with open(os.path.join(config.outputs_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    best_model = min(results.items(), key=lambda kv: kv[1]["RMSE"])[0]
    best_pred = preds_hold[best_model]
    out_df = test_df[["country", "year"]].copy()
    out_df["y_true"] = y_test
    out_df["y_pred"] = best_pred
    out_df["model"] = best_model
    out_df.to_csv(os.path.join(config.outputs_dir, "pred_vs_actual_test.csv"), index=False)
    plt.figure(figsize=(6,6))
    plt.scatter(out_df["y_true"], out_df["y_pred"], alpha=0.7)
    lims = [min(out_df["y_true"].min(), out_df["y_pred"].min()), max(out_df["y_true"].max(), out_df["y_pred"].max())]
    plt.plot(lims, lims, "--")
    plt.xlabel("Actual CO₂ (Mt)")
    plt.ylabel("Predicted CO₂ (Mt)")
    plt.title(f"Predicted vs Actual ({best_model})")
    plt.tight_layout()
    plt.savefig(os.path.join(config.outputs_dir, "pred_vs_actual_plot.png"), dpi=200)
    plt.close()
    return results
