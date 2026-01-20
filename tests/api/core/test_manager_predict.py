import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.api.core.manager import PipelineManager


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.feature_names = ["size", "rooms", "has_garden"]
    model.predict.return_value = np.array([np.log1p(300000)])
    return model


@pytest.fixture
def manager(mock_model):
    manager = PipelineManager()
    manager._initialized = True
    manager.model = mock_model
    return manager


def test_predict_happy_path(manager):
    features = {
        "size": 120,
        "rooms": 4,
        "has_garden": 1,
    }

    result = manager.predict(features)

    assert result["success"] is True
    assert pytest.approx(result["prediction"], rel=1e-6) == 300000
    assert result["error"] is None
    assert set(result["features"].keys()) == {"size", "rooms", "has_garden"}


def test_predict_missing_features_filled(manager):
    features = {
        "size": 100,  # missing rooms, has_garden
    }

    result = manager.predict(features)

    assert result["success"] is True
    assert result["features"]["rooms"] == 0
    assert result["features"]["has_garden"] == 0


def test_predict_extra_features_ignored(manager):
    features = {
        "size": 100,
        "rooms": 3,
        "has_garden": 1,
        "random_noise_feature": 999,
    }

    result = manager.predict(features)

    assert result["success"] is True
    assert "random_noise_feature" not in result["features"]


def test_predict_drops_non_numeric_columns(manager):
    features = {
        "size": 80,
        "rooms": 2,
        "has_garden": 1,
        "description": "nice house",
    }

    result = manager.predict(features)

    assert result["success"] is True
    assert "description" not in result["features"]


def test_predict_fails_without_model_feature_names():
    manager = PipelineManager()
    manager._initialized = True

    bad_model = MagicMock()
    bad_model.feature_names = None
    manager.model = bad_model

    result = manager.predict({"size": 100})

    assert result["success"] is False
    assert result["prediction"] is None
    assert "model does not expose feature names" in result["error"].lower()


def test_predict_not_initialized():
    manager = PipelineManager()
    manager._initialized = False

    with pytest.raises(RuntimeError):
        manager.predict({"size": 100})
