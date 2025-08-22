import re
import pandas as pd
import numpy as np


def to_int(s):
    """
    Extract digits and convert to int, allowing thousand separators.
    """
    if s is None or pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip()
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else np.nan


def to_float_pct(s):
    """
    Convert percentage string to float (0-1).
    """
    if s is None or pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s) / 100
    s = str(s).strip()
    digits = re.sub(r"[^\d]", "", s)
    return float(digits)/100 if digits else np.nan


def safe_to_float_currency(s):
    """
    Convert currency string to float.
    """
    if s is None or pd.isna(s):
        return np.nan

    s = str(s).strip()
    if not s:
        return np.nan

    # Remove everything except digits and dots
    s = re.sub(r"[^\d.]", "", s)

    # Keep only the first dot (merge the rest)
    if s.count('.') > 1:
        parts = s.split('.')
        s = parts[0] + '.' + ''.join(parts[1:])

    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_price(s):
    """
    Convert price string to int, allowing currency symbols
    and thousand separators.
    """
    if s is None or pd.isna(s):
        return np.nan
    s = str(s).strip()
    digits = re.sub(r"[^\d]", "", s) 
    return int(digits) if digits else np.nan


def parse_size(s):
    """
    Convert size string to numeric (int), allowing unit suffixes
    and thousand separators. Returns np.nan for missing/invalid values.
    """
    if s is None or pd.isna(s):
        return np.nan
    digits = re.sub(r"[^\d]", "", str(s))
    return int(digits) if digits else np.nan


def split_postal_city(s):
    """
    Split postal code and city from a string like '1012 AB Amsterdam'.
    """
    if not isinstance(s, str):
        return pd.NA, pd.NA
    parts = s.strip().rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return s, pd.NA
