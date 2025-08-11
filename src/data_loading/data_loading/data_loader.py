import glob
import json
import pandas as pd

def load_data_from_json(path_pattern):
    files = glob.glob(path_pattern)
    data_list = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data_list.append(json.load(f))
    return pd.DataFrame(data_list)