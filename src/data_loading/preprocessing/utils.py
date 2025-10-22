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
    return float(digits) / 100 if digits else np.nan


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
    if s.count(".") > 1:
        parts = s.split(".")
        s = parts[0] + "." + "".join(parts[1:])

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
    Returns (postal_code_clean, city)
    """

    if not isinstance(s, str):
        return pd.NA, pd.NA

    s = s.strip().upper()  # uppercase and remove extra spaces
    match = re.match(r"(\d{4}\s?[A-Z]{2})\s*(.*)", s)
    if match:
        postal_code = match.group(1)
        city = match.group(2) if match.group(2) else pd.NA
        return postal_code, city

    return s, pd.NA


def parse_year(year):
    """
    Convert year string to int, allowing non-digit characters.
    """
    if isinstance(year, str):
        if year.startswith("Voor"):  # e.g., "Voor 1906"
            return int(year.split()[-1]) - 1  # use 1905
        elif year.startswith("Na"):  # e.g., "Na 2020"
            return int(year.split()[-1]) + 1  # use 2021
        elif year.isdigit():
            return int(year)
        else:
            return np.nan
    elif isinstance(year, (int, float)):
        return int(year)
    else:
        return np.nan


def coerce_numeric(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce")


def apply_parsers(df, col_parser_map):
    for col, parser in col_parser_map.items():
        df[col + "_num"] = df[col].apply(parser)
    return df
