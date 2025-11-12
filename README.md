# CO₂ Forecast from Energy Use — Regression Baseline

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
