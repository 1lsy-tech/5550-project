# CO₂ Forecast from Energy Use — Regression Baseline

<<<<<<< HEAD
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

=======
>>>>>>> 31cbd61a6b0eea627889766f887c47eb416f2871
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
