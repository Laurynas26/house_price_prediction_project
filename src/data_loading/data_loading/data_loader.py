import glob
import json
import pandas as pd


def load_data_from_json(path_pattern: str) -> pd.DataFrame:
    """
    Load multiple JSON files matching a path pattern into a single DataFrame.

    Parameters
    ----------
    path_pattern : str
        Glob-style file path pattern to match JSON files
        (e.g., "data/raw/*.json").

    Returns
    -------
    pd.DataFrame
        DataFrame where each row corresponds to one JSON file's content.

    Notes
    -----
    - Assumes each JSON file contains a dictionary-like structure.
    - If JSON files contain nested structures, further normalization may be required.
    - Files are read using UTF-8 encoding.
    """
    files = glob.glob(path_pattern)
    data_list = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data_list.append(json.load(f))
    return pd.DataFrame(data_list)
