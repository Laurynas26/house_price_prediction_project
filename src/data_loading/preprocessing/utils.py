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
        return float(digits) if digits else pd.NA
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