import re
import pandas as pd

def to_int(s):
    """
    Extract digits and convert to int, allowing thousand separators.
    """
    if s is None:
        return pd.NA
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip()
    if not s:
        return pd.NA
    # Remove everything except digits
    digits = re.sub(r"[^\d]", "", s)
    try:
        return int(digits) if digits else pd.NA
    except ValueError:
        return pd.NA


def to_float_pct(s):
    """
    Convert percentage string to float (0-100).
    """
    if s is None:
        return pd.NA
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s:
        return pd.NA
    digits = re.sub(r"[^\d]", "", s)
    try:
        return float(digits) / 100 if digits else pd.NA
    except ValueError:
        return pd.NA

def safe_to_float_currency(s):
    """
    Convert currency to float (0-100).
    """
    if s is None:
        return pd.NA
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s:
        return pd.NA
    digits = re.sub(r"[^\d]", "", s)
    try:
        return float(digits) if digits else pd.NA
    except ValueError:
        return pd.NA
    
def parse_price(price_str):
    """
    Convert price string to int, allowing currency symbols and thousand separators.
    """
    if pd.isna(price_str):
        return pd.NA
    price_str = re.sub(r"[^\d]", "", price_str)
    return int(price_str) if price_str else pd.NA


def parse_size(size_str):
    """"
    Convert size string to int, allowing unit suffixes and thousand separators.
    """
    if pd.isna(size_str):
        return pd.NA
    digits = re.sub(r"[^\d]", "", size_str)
    return int(digits) if digits else pd.NA

def split_postal_city(postal_city_str):
    """
    Split postal code and city from a string like '1012 AB Amsterdam'.
    """
    if not isinstance(postal_city_str, str):
        return pd.NA, pd.NA
    parts = postal_city_str.strip().rsplit(' ', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return postal_city_str, pd.NA