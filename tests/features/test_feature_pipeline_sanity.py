import numpy as np
from pathlib import Path
import pandas as pd

from src.features.preprocessing_pipeline import PreprocessingPipeline


def test_feature_pipeline_sanity(tmp_path):
    """
    Sanity test for full preprocessing + feature engineering pipeline.

    Ensures:
    - X matrices are non-empty
    - All features are numeric
    - No NaNs are present
    - Feature columns are stable and ordered
    """

    # --- Fake dataset ---
    df = pd.DataFrame(
        {
            "size_num": [50, 70, 100],
            "rooms": [2, 3, 4],
            "has_garden": [1, 0, 1],
            "price": [200000, 300000, 350000],  # include target
        }
    )

    # --- Minimal config ---
    config_paths = {
        "preprocessing": {
            "drop_raw": True,
            "numeric_cols": ["size_num", "rooms", "has_garden"],
            "imputation": {},
        }
    }

    pipeline = PreprocessingPipeline(
        config_paths=config_paths,
        raw_json_pattern=None,  # disable reading JSON
        model_config_path=Path("config/model_config.yaml"),
        model_name="dummy_model",
        load_cache=False,
        save_cache=False,
    )

    # Patch the run method to use our fake df
    pipeline._load_raw_data = lambda: df  # monkey-patch

    result = pipeline.run(smart_cache=False)

    # --- Assertions ---
    assert result.X_train is not None
    assert not result.X_train.empty
    assert result.X_train.shape[0] == 3
    assert all(
        np.issubdtype(dtype, np.number) for dtype in result.X_train.dtypes
    )
    assert result.X_train.isnull().sum().sum() == 0
    assert pipeline.expected_columns == result.X_train.columns.tolist()
