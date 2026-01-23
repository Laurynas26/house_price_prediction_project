import numpy as np
from pathlib import Path

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

    # --- Minimal config paths ---
    config_paths = {
        "preprocessing": {
            "drop_raw": True,
            "numeric_cols": [],
            "imputation": {},
        }
    }

    pipeline = PreprocessingPipeline(
        config_paths=config_paths,
        raw_json_pattern="data/parsed_json/*.json",
        model_config_path=Path("config/model_config.yaml"),
        model_name="xgboost_extended",  # use a real model name
        load_cache=False,
        save_cache=False,
    )

    result = pipeline.run(smart_cache=False)

    # --- Basic existence ---
    assert result.X_train is not None
    assert not result.X_train.empty

    # --- Shape sanity ---
    assert result.X_train.shape[0] > 0
    assert result.X_train.shape[1] > 5  # protects against accidental drops

    # --- Numeric-only ---
    assert all(
        np.issubdtype(dtype, np.number) for dtype in result.X_train.dtypes
    )

    # --- No NaNs ---
    assert result.X_train.isnull().sum().sum() == 0

    # --- Schema consistency ---
    assert pipeline.expected_columns == result.X_train.columns.tolist()
