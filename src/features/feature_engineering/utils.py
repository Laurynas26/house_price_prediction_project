import re
import numpy as np
import pandas as pd


def to_float(value):
    """Convert strings with units/symbols into floats."""
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[^\d\.]", "", str(value))
    return float(cleaned) if cleaned else np.nan


def extract_floor(x):
    """Extract floor number from string like '3e verdieping'."""
    if pd.isna(x) or x in ["N/A", "Begane grond"]:
        return 0
    match = re.search(r"(\d+)", str(x))
    return int(match.group(1)) if match else 0


def extract_lease_years(x, current_year=2025):
    """Extract lease years remaining from ownership text."""
    if pd.isna(x) or "Volle eigendom" in str(x) or str(x).strip() == "":
        return np.nan
    match = re.search(r"einddatum erfpacht: (\d{2})-(\d{2})-(\d{4})", str(x))
    if match:
        _, _, year = map(int, match.groups())
        return max(year - current_year, 0)
    return np.nan


def drop_low_variance_dummies(df, threshold=0.95):
    """Drop categorical dummy columns with low variance."""
    low_var_cols = [
        col
        for col in df.columns
        if df[col].value_counts(normalize=True, dropna=False).iloc[0] >= threshold
    ]
    return df.drop(columns=low_var_cols), low_var_cols
