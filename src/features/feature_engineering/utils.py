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
    """Extract floor number from string like '3e verdieping', safely handling non-scalars."""
    # If x is a Series, list, or array, take the first element
    if isinstance(x, (pd.Series, list, np.ndarray)):
        if len(x) > 0:
            x = x[0]
        else:
            return 0

    if pd.isna(x) or x in ["N/A", "Begane grond"]:
        return 0

    try:
        match = re.search(r"(\d+)", str(x))
        return int(match.group(1)) if match else 0
    except Exception:
        return 0


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
        if df[col].value_counts(normalize=True, dropna=False).iloc[0]
        >= threshold
    ]
    return df.drop(columns=low_var_cols), low_var_cols


def auto_log_transform_train(df_train, numeric_cols, threshold_skew=0.5):
    """Compute which columns to log-transform on training fold and return fitted info."""
    log_cols = []
    for col in numeric_cols:
        if (df_train[col] > 0).all():
            skewness = df_train[col].skew()
            if abs(skewness) > threshold_skew:
                df_train[f"log_{col}"] = np.log1p(df_train[col])
                log_cols.append(f"log_{col}")
    return df_train, log_cols


def apply_log_transform(df, log_cols):
    """Apply log1p transform using training fold columns."""
    for col in log_cols:
        orig_col = col.replace("log_", "")
        df[col] = np.log1p(df[orig_col])
    return df


def simplify_roof(roof):
    if pd.isna(roof) or str(roof) == "N/A":
        return "Unknown"
    roof = str(roof)
    if "Plat dak" in roof:
        return "Flat"
    if "Zadeldak" in roof:
        return "Saddle"
    if "Samengesteld dak" in roof:
        return "Composite"
    if "Mansarde" in roof:
        return "Mansard"
    return "Other"


def simplify_ownership(x):
    if pd.isna(x) or str(x).strip() == "":
        return "Unknown"
    x = str(x)
    if "Volle eigendom" in x:
        return "Full"
    if "Erfpacht" in x and "Gemeentelijk" in x:
        return "Municipal"
    if "Erfpacht" in x:
        return "Leasehold"
    return "Other"


def simplify_location(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x)
    if "centrum" in x:
        return "Central"
    if "woonwijk" in x:
        return "Residential"
    if "vrij uitzicht" in x:
        return "OpenView"
    if "park" in x:
        return "Park"
    return "Other"
