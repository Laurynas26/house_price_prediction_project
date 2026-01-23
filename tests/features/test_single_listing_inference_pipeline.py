from src.features.preprocessing_pipeline import PreprocessingPipeline
from pathlib import Path


def test_single_listing_feature_schema_matches_training():

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
        model_name="xgboost_extended",
        load_cache=False,
        save_cache=False,
    )

    pipeline.run(smart_cache=False)

    # --- Minimal realistic listing ---
    listing = {
        "price": "€ 400.000",
        "size": "80 m²",
        "nr_rooms": 3,
        "postal_code": "1011 AB",
        "energy_label": "A",
        "ownership_type": "Freehold",
        "status": "Available",
        "roof_type": "Flat",
        "location": "In woonwijk",
        "garden": None,
        "address": "Teststraat 1",
    }

    X_inf = pipeline.transform_single_for_inference(listing, drop_target=True)

    # --- Schema contract ---
    assert list(X_inf.columns) == pipeline.expected_columns
    assert X_inf.shape[0] == 1

    # --- Numeric-only ---
    assert all(np.issubdtype(dtype, np.number) for dtype in X_inf.dtypes)

    # --- No NaNs ---
    assert X_inf.isnull().sum().sum() == 0
