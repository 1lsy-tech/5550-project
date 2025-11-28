"""
Utilities for downloading, cleaning and sub-sampling the OWID + World Bank data.

This module is not used in the main training pipeline at grading time, but it
documents how the sample dataset `data/sample_owid_energy_co2.csv` can be created
from a larger merged table.

The idea is:
1. Start from a "full" merged table that combines OWID CO2 + energy data with
   World Bank GDP per capita and population.
2. Clean the table (drop missing values, keep relevant columns).
3. Optionally filter to a subset of countries / years to keep the project lightweight.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

# Paths
DATA_DIR = Path("data")
# User-supplied merged file that combines OWID + World Bank data
RAW_FULL_CSV = DATA_DIR / "owid_energy_co2_full.csv"
# Final sample used by the modelling pipeline
SAMPLE_CSV = DATA_DIR / "sample_owid_energy_co2.csv"

# Columns expected by the modelling pipeline
FEATURE_COLUMNS: List[str] = [
    "country",
    "year",
    "co2_mt",
    "coal_consumption_twh",
    "oil_consumption_twh",
    "gas_consumption_twh",
    "renewables_consumption_twh",
    "gdp_per_capita_usd",
    "population",
]


def fetch_and_merge_raw_data() -> pd.DataFrame:

    if not RAW_FULL_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_FULL_CSV} not found.\n"
            "Please download / build the full merged OWID + World Bank table and "
            "save it as 'data/owid_energy_co2_full.csv'. See README 'Data Sources' "
            "for the original URLs and preprocessing description."
        )
    df = pd.read_csv(RAW_FULL_CSV)
    return df


def clean_and_save_sample(
    min_year: int = 1990,
    max_year: int = 2022,
    focus_countries: Optional[List[str]] = None,
    out_path: Path = SAMPLE_CSV,
) -> pd.DataFrame:

    df = fetch_and_merge_raw_data()

    # Keep only the columns needed by the modelling pipeline
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The full dataset is missing required columns: {missing_cols}. "
            "Please make sure the merged table contains these variables."
        )
    df = df[FEATURE_COLUMNS].copy()

    # Filter by year
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]

    # Optionally filter by country
    if focus_countries is not None:
        df = df[df["country"].isin(focus_countries)]

    # Drop rows with any missing values in the modelling features
    df = df.dropna(subset=FEATURE_COLUMNS)

    # Sort for readability: by country, then by year
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    # Save to disk – this is the exact file consumed by `Config.data_csv`
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned sample dataset to: {out_path.resolve()}")

    return df


if __name__ == "__main__":
    # Example usage:
    # For a lightweight sample, we can restrict to G20-like economies.
    g20_example = [
        "United States",
        "China",
        "India",
        "European Union",
        "Japan",
        "United Kingdom",
        "Germany",
        "France",
        "Italy",
        "Canada",
        "Brazil",
        "Russia",
        "Australia",
        "South Africa",
        "Mexico",
        "South Korea",
        "Indonesia",
        "Turkey",
        "Saudi Arabia",
        "Argentina",
    ]

    clean_and_save_sample(
        min_year=1990,
        max_year=2022,
        focus_countries=g20_example,
    )
