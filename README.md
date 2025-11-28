# CO₂ Forecast from Energy Use — Regression Baseline

## Problem Definition

This project aims to predict annual country-level CO₂ emissions using a combination of macro-level indicators, including energy consumption structure, GDP per capita, and population. Understanding the drivers of national CO₂ emissions is important for climate policy, long-term planning, and evaluating the potential impacts of energy or economic transitions. By modeling CO₂ emissions with a set of transparent and interpretable features, the project seeks to highlight which socioeconomic and energy variables contribute most to changes in emissions.
The overarching goal is to build a simple, explainable, and fully reproducible machine-learning pipeline that can be run end-to-end and easily compared across multiple baseline models.

## Data Sources

The dataset used in this project is derived from two widely recognized open-data providers: Our World in Data (OWID) and the World Bank Open Data platform.
OWID provides detailed, country-level historical records on CO₂ emissions and energy consumption, including metrics such as fossil fuel use, electricity generation, and primary energy shares. The World Bank offers complementary socioeconomic indicators such as GDP per capita and population, which are essential explanatory variables for modeling national emissions.
For reproducibility, this repository includes a pre-processed subset of the original datasets (sample_owid_energy_co2.csv). This file was created by cleaning, merging, and selecting relevant features from the full OWID and World Bank datasets. The preprocessing removes missing values, aligns country-year pairs, and retains only the variables necessary for regression modeling.
Links to the original data sources are provided below:

Our World in Data – CO₂ and Greenhouse Gas Emissions
https://ourworldindata.org/co2-and-greenhouse-gas-emissions

Our World in Data – Energy Dataset
https://ourworldindata.org/energy

World Bank Open Data (GDP & Population Indicators)
https://data.worldbank.org/

This ensures transparency regarding data origin and allows the full pipeline to be reproduced using either the included sample dataset or the original online sources.

The file `data/sample_owid_energy_co2.csv` was generated using `src/data_prep.py`
from a larger merged OWID + World Bank dataset (`data/owid_energy_co2_full.csv`)
by selecting the relevant features, filtering years, optionally focusing on a
subset of countries (e.g., G20 economies), and dropping rows with missing values.

## Methods and Implementation

This project formulates national CO₂ forecasting as a supervised regression task.
The target variable is annual country-level CO₂ emissions (in megatonnes), and
the features include energy consumption by source (coal, oil, gas, renewables),
GDP per capita, and population.

We implement a small family of baseline and slightly more expressive models:

- Multiple Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

To avoid overfitting to a single train–test split and to respect the temporal
ordering of the data, we use **time-series cross-validation** with
`TimeSeriesSplit`. For Ridge, Lasso, and Random Forest we perform a simple
grid search over a small set of hyperparameters (e.g., different `alpha` values
for Ridge/Lasso and different `n_estimators` / `max_depth` values for the
Random Forest). For each candidate configuration we compute the mean MAE, RMSE,
and R² across the cross-validation folds, and select the setting with the
lowest average RMSE.

After tuning, the best configuration of each model is refit on the full training
period and evaluated on a held-out test period (later years). The final metrics
are saved in `outputs/metrics.json`, and the best hyperparameters and CV scores
are logged in `outputs/best_hyperparameters.json` for transparency and
reproducibility.

This is a **minimal, working, end-to-end** pipeline that:
1. Reads and refines energy + CO₂ data (synthetic sample provided; can fetch OWID in Colab).
2. Trains baseline regressors (Linear, Ridge, Lasso, Random Forest).
3. Evaluates on a hold-out set (by **year** to mimic forecasting).
4. Produces metrics and plots in the `outputs/` folder.

> Use `run_pipeline.py` for a one-click run (works offline with the bundled sample).  

## Quickstart (local or Colab)

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Artifacts will appear in `outputs/`:
- `metrics.json` — MAE / RMSE / R² for each model
- `pred_vs_actual_test.csv` — predictions vs. ground truth
- `pred_vs_actual_plot.png` — scatter plot
