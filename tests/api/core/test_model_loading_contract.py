import numpy as np
from unittest.mock import MagicMock, patch
import pickle

from src.api.core.manager import PipelineManager

GEO_PATCH = (
    "src.features.feature_engineering."
    "location_feature_enrichment.enrich_with_geolocation"
)

AMENITY_PATCH = (
    "src.features.feature_engineering."
    "location_feature_enrichment.enrich_with_amenities"
)


def test_pipeline_manager_model_loading_contract(tmp_path):
    """
    Contract test:
    - load_production_model returns a model with required interface
    - PipelineManager initializes successfully
    - Prediction path is callable
    """

    # --- Reset singleton safely ---
    PipelineManager._instance = None

    # --- Fake inference_meta.pkl (shape-accurate) ---
    inference_meta = {
        "meta": {
            "geo_meta": {
                "cache_file": "dummy.csv",
                "postal_code_centroids": {},
                "neighborhood_centroids": {},
            },
            "amenity_meta": {
                "amenity_radius_map": {},
                "encoded_columns": [],
            },
        },
        "expected_columns": ["size_num", "rooms", "has_garden"],
    }

    config_dir = tmp_path
    (config_dir / "inference_meta.pkl").write_bytes(
        pickle.dumps(inference_meta)
    )

    (config_dir / "preprocessing_config.yaml").write_text("dummy: true")
    (config_dir / "model_config.yaml").write_text(
        "production_model_name: dummy_model"
    )

    # --- Mock model ---
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([np.log1p(350000)])

    with patch(
        "src.api.core.manager.load_production_model",
        return_value=mock_model,
    ), patch(
        GEO_PATCH,
        side_effect=lambda df, **kw: (df, kw.get("geo_meta")),
    ), patch(
        AMENITY_PATCH,
        side_effect=lambda df, **kw: (df, kw.get("amenity_meta")),
    ):

        manager = PipelineManager()
        manager.initialize(str(config_dir))

        # --- Assertions ---
        assert manager._initialized is True
        assert manager.model is mock_model
        assert manager.pipeline.expected_columns == [
            "size_num",
            "rooms",
            "has_garden",
        ]

        # --- Prediction sanity ---
        result = manager.predict(
            {"size_num": 100, "rooms": 3, "has_garden": 1}
        )

        assert result["success"] is True
        assert abs(result["prediction"] - 350000) < 1e-6
