import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.api.core.manager import PipelineManager


def test_pipeline_manager_model_loading_contract(tmp_path):
    """
    Contract test:
    - load_production_model returns a model with required interface
    - PipelineManager initializes successfully
    - Prediction path is callable
    """

    # --- Fake inference_meta.pkl ---
    inference_meta = {
        "meta": {},
        "expected_columns": ["size_num", "rooms", "has_garden"],
    }

    config_dir = tmp_path
    (config_dir / "inference_meta.pkl").write_bytes(
        __import__("pickle").dumps(inference_meta)
    )

    # --- Minimal configs ---
    (config_dir / "preprocessing_config.yaml").write_text("dummy: true")
    (config_dir / "model_config.yaml").write_text(
        """
production_model_name: dummy_model
experiment_name: dummy_experiment
"""
    )

    # --- Mock model ---
    mock_model = MagicMock()
    mock_model.feature_names = ["size_num", "rooms", "has_garden"]
    mock_model.predict.return_value = np.array([np.log1p(350000)])

    with patch(
        "src.api.core.manager.load_production_model",
        return_value=mock_model,
    ):
        manager = PipelineManager()
        manager._instance = None  # reset singleton for test
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
