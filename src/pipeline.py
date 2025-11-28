import os, json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
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

def plot_country_timeseries(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    countries: list[str],
    out_dir: str,
):
    """
    Plot actual vs predicted CO₂ for selected countries across test years.
    """
    df_plot = test_df.copy()
    df_plot["y_true"] = y_true
    df_plot["y_pred"] = y_pred

    for c in countries:
        sub = df_plot[df_plot["country"] == c]
        if len(sub) == 0:
            continue

        plt.figure(figsize=(7, 4))
        plt.plot(sub["year"], sub["y_true"], label="Actual", marker="o")
        plt.plot(sub["year"], sub["y_pred"], label="Predicted", marker="o")
        plt.title(f"{c}: CO₂ emissions (Actual vs Predicted)")
        plt.xlabel("Year")
        plt.ylabel("CO₂ (Mt)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"time_series_{c.replace(' ', '_')}.png"), dpi=200)
        plt.close()


def plot_metric_bars(results: Dict[str, Dict[str, float]], out_dir: str) -> None:

    models = list(results.keys())
    maes = [results[m]["MAE"] for m in models]
    rmses = [results[m]["RMSE"] for m in models]
    r2s = [results[m]["R2"] for m in models]

    # MAE bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(models, maes)
    plt.ylabel("MAE")
    plt.title("MAE by model (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_mae_bar.png"), dpi=200)
    plt.close()

    # RMSE bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(models, rmses)
    plt.ylabel("RMSE")
    plt.title("RMSE by model (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_rmse_bar.png"), dpi=200)
    plt.close()

    # R2 bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(models, r2s)
    plt.ylabel("R²")
    plt.title("R² by model (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_r2_bar.png"), dpi=200)
    plt.close()


def plot_feature_importances(
    models: Dict[str, Any],
    feature_names: list[str],
    out_dir: str,
) -> None:

    # --- RandomForest feature importances ---
    if "RandomForest" in models and hasattr(models["RandomForest"], "feature_importances_"):
        rf = models["RandomForest"]
        importances = rf.feature_importances_
        idx_sorted = np.argsort(importances)[::-1]  # descending
        sorted_names = [feature_names[i] for i in idx_sorted]
        sorted_imp = importances[idx_sorted]

        plt.figure(figsize=(7, 5))
        plt.barh(sorted_names, sorted_imp)
        plt.xlabel("Importance")
        plt.title("RandomForest feature importances")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importance_random_forest.png"), dpi=200)
        plt.close()

    # --- Linear coefficients (use Ridge if available, else LinearRegression) ---
    linear_key = None
    if "Ridge" in models:
        linear_key = "Ridge"
    elif "LinearRegression" in models:
        linear_key = "LinearRegression"

    if linear_key is not None and hasattr(models[linear_key], "coef_"):
        lm = models[linear_key]
        coefs = lm.coef_
        idx_sorted = np.argsort(np.abs(coefs))[::-1]
        sorted_names = [feature_names[i] for i in idx_sorted]
        sorted_coefs = coefs[idx_sorted]

        plt.figure(figsize=(7, 5))
        plt.barh(sorted_names, sorted_coefs)
        plt.xlabel("Coefficient")
        plt.title(f"{linear_key} coefficients (sorted by absolute value)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"feature_coefficients_{linear_key}.png"), dpi=200)
        plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: str,
) -> None:
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted CO₂ (Mt)")
    plt.ylabel("Residual (true - pred)")
    plt.title(f"Residuals vs predicted ({model_name})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def cross_val_model(model, X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 3) -> Dict[str, float]:

    tscv = TimeSeriesSplit(n_splits=n_splits)

    mae_list, rmse_list, r2_list = [], [], []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # clone() makes a fresh copy of the model for each fold
        m = clone(model)
        m.fit(X_tr, y_tr)
        y_val_pred = m.predict(X_val)

        m_metrics = metrics(y_val, y_val_pred)
        mae_list.append(m_metrics["MAE"])
        rmse_list.append(m_metrics["RMSE"])
        r2_list.append(m_metrics["R2"])

    return {
        "MAE": float(np.mean(mae_list)),
        "RMSE": float(np.mean(rmse_list)),
        "R2": float(np.mean(r2_list)),
    }


def grid_search_time_series(
    model_class,
    param_grid: Dict[str, list],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    base_kwargs: Optional[Dict[str, Any]] = None,
):


    from itertools import product

    if base_kwargs is None:
        base_kwargs = {}

    best_rmse = float("inf")
    best_params: Dict[str, Any] = {}
    best_cv_metrics: Dict[str, float] | None = None

    # Create all combinations of hyperparameters
    keys = list(param_grid.keys())
    values_list = [param_grid[k] for k in keys]

    for values in product(*values_list):
        params = dict(zip(keys, values))
        full_params = {**base_kwargs, **params}

        model = model_class(**full_params)
        cv_metrics = cross_val_model(model, X_train, y_train, n_splits=n_splits)

        if cv_metrics["RMSE"] < best_rmse:
            best_rmse = cv_metrics["RMSE"]
            best_params = full_params
            best_cv_metrics = cv_metrics

    return best_params, best_cv_metrics


def run_all(config: Config) -> Dict[str, Dict[str, float]]:

    maybe_makedirs(config.outputs_dir)

    # ---------- 1. Load data and split ----------
    df = load_data(config.data_csv)
    (X_train, y_train), (X_test, y_test), train_df, test_df = train_test_split_by_year(
        df, config.test_start_year, config.target_col, config.feature_cols
    )

    # ---------- BASELINE 1: naive last-year carry-forward ----------
    baseline1_preds = []
    for i, row in test_df.iterrows():
        country = row["country"]
        year = row["year"]
        # Find the country's CO₂ for the previous year
        prev = df[(df["country"] == country) & (df["year"] == year - 1)]
        if len(prev) > 0:
            baseline1_preds.append(prev[config.target_col].values[0])
        else:
            # If the previous year's data is unavailable, use the average value of the training set.
            baseline1_preds.append(float(np.mean(y_train)))

    baseline1_metrics = metrics(y_test, np.array(baseline1_preds))

    # ---------- BASELINE 2: GDP-only linear regression ----------
    from sklearn.linear_model import LinearRegression as SKLinearRegression

    gdp_train = train_df[["gdp_per_capita_usd"]].values
    gdp_test = test_df[["gdp_per_capita_usd"]].values

    gdp_model = SKLinearRegression()
    gdp_model.fit(gdp_train, y_train)
    gdp_pred = gdp_model.predict(gdp_test)

    baseline2_metrics = metrics(y_test, gdp_pred)


    # ---------- 2. Hyperparameter tuning with TimeSeriesSplit ----------
    # LinearRegression: no hyperparameters to tune in this simple setup
    linear_model = LinearRegression()

    # Ridge: alpha in {0.1, 1.0, 10.0}
    ridge_param_grid = {"alpha": [0.1, 1.0, 10.0]}
    ridge_best_params, ridge_cv_metrics = grid_search_time_series(
        Ridge,
        ridge_param_grid,
        X_train,
        y_train,
        n_splits=3,
        base_kwargs={},  # can add "random_state" if desired for other models
    )
    ridge_model = Ridge(**ridge_best_params)

    # Lasso: alpha in {1e-4, 1e-3, 1e-2}
    lasso_param_grid = {"alpha": [1e-4, 1e-3, 1e-2]}
    lasso_best_params, lasso_cv_metrics = grid_search_time_series(
        Lasso,
        lasso_param_grid,
        X_train,
        y_train,
        n_splits=3,
        base_kwargs={"max_iter": 10000, "random_state": 42},
    )
    lasso_model = Lasso(**lasso_best_params)

    # RandomForest: n_estimators in {100, 300}, max_depth in {None, 5, 10}
    rf_param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 5, 10],
    }
    rf_best_params, rf_cv_metrics = grid_search_time_series(
        RandomForestRegressor,
        rf_param_grid,
        X_train,
        y_train,
        n_splits=3,
        base_kwargs={"random_state": 42, "n_jobs": -1},
    )
    rf_model = RandomForestRegressor(**rf_best_params)

    # Save best hyperparameters & CV metrics for transparency
    hp_summary = {
        "Ridge": {"best_params": ridge_best_params, "cv_metrics": ridge_cv_metrics},
        "Lasso": {"best_params": lasso_best_params, "cv_metrics": lasso_cv_metrics},
        "RandomForest": {"best_params": rf_best_params, "cv_metrics": rf_cv_metrics},
    }

    with open(os.path.join(config.outputs_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(hp_summary, f, indent=2)


    # ---------- 3. Fit all models on full training data ----------
    models = {
        "LinearRegression": linear_model,
        "Ridge": ridge_model,
        "Lasso": lasso_model,
        "RandomForest": rf_model,
    }

    results: Dict[str, Dict[str, float]] = {}
    preds_hold: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = metrics(y_test, y_pred)
        preds_hold[name] = y_pred

    # ---------- 4. Save metrics, predictions and plots ----------
    # Save metrics
    # add baselines into results
    results["Baseline_LastYear"] = baseline1_metrics
    results["Baseline_GDPonly"] = baseline2_metrics
    with open(os.path.join(config.outputs_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Extra visualization: bar charts for MAE / RMSE / R2
    plot_metric_bars(results, config.outputs_dir)

    # Extra visualization: feature importance / coefficients
    plot_feature_importances(models, config.feature_cols, config.outputs_dir)

    # choose best model on test RMSE
    best_model = min(results.items(), key=lambda kv: kv[1]["RMSE"])[0]
    best_pred = preds_hold[best_model]

    out_df = test_df[["country", "year"]].copy()
    out_df["y_true"] = y_test
    out_df["y_pred"] = best_pred
    out_df["model"] = best_model
    out_df.to_csv(os.path.join(config.outputs_dir, "pred_vs_actual_test.csv"), index=False)

    # Predicted vs actual scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(out_df["y_true"], out_df["y_pred"], alpha=0.7)
    lims = [
        min(out_df["y_true"].min(), out_df["y_pred"].min()),
        max(out_df["y_true"].max(), out_df["y_pred"].max()),
    ]
    plt.plot(lims, lims, "--")
    plt.xlabel("Actual CO₂ (Mt)")
    plt.ylabel("Predicted CO₂ (Mt)")
    plt.title(f"Predicted vs Actual ({best_model})")
    plt.tight_layout()
    plt.savefig(os.path.join(config.outputs_dir, "pred_vs_actual_plot.png"), dpi=200)
    plt.close()

    # Residual plot for the best model
    residual_plot_path = os.path.join(config.outputs_dir, "residuals_best_model.png")
    plot_residuals(y_test, best_pred, best_model, residual_plot_path)

    # Time-series plots for selected countries
    plot_country_timeseries(
        test_df,
        y_test,
        best_pred,
        countries=["United States", "China", "European Union"],
        out_dir=config.outputs_dir,
    )


    return results


