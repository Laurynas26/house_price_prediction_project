from pathlib import Path
from typing import Dict, Any
import yaml
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle

from src.scraper.core import scrape_listing
from src.features.preprocessing_pipeline import (
    PreprocessingPipeline,
)
from src.api.core.mlflow_utils import load_production_model
from src.api.core.schemas import build_listing_from_manual_input

from src.features.data_prep_for_modelling.data_preparation import (
    load_geo_config,
)

from src.features.feature_engineering.location_feature_enrichment import (
    load_cache,
)

import logging

logger = logging.getLogger(__name__)


RAW_JSON_PATTERN = Path(__file__).parents[3] / "data/parsed_json/*.json"

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX")


class PipelineManager:
    """
    Singleton manager for real estate pipeline:
    - Loads and runs preprocessing pipeline
    - Loads trained model
    - Handles scraping → preprocessing → prediction in a consistent way
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _load_configs(self, config_dir: Path) -> tuple[dict, dict]:
        """
        Load preprocessing and model configuration files.

        Args:
            config_dir: Directory containing YAML configuration files.

        Returns:
            Tuple of (preprocessing_cfg, model_cfg) dictionaries.
        """
        logger.info("Loading preprocessing and model configs")

        with open(config_dir / "preprocessing_config.yaml") as f:
            preprocessing_cfg = yaml.safe_load(f)

        with open(config_dir / "model_config.yaml") as f:
            model_cfg = yaml.safe_load(f)

        return preprocessing_cfg, model_cfg

    def _detect_runtime(self) -> dict:
        """
        Detect runtime environment and return environment-specific settings.

        Returns:
            Dictionary containing runtime flags
            such as S3 usage and cache behavior.
        """
        running_on_lambda = "AWS_LAMBDA_FUNCTION_NAME" in os.environ

        if running_on_lambda:
            logger.info("Detected AWS Lambda environment")
            return {
                "use_s3": True,
                "raw_json_pattern": None,
                "load_pipeline_cache": True,
                "save_pipeline_cache": False,
            }

        logger.info("Detected local environment")
        return {
            "use_s3": False,
            "raw_json_pattern": None,
            "load_pipeline_cache": False,
            "save_pipeline_cache": False,
        }

    def _build_preprocessing_pipeline(
        self,
        preprocessing_cfg: dict,
        model_cfg: dict,
        runtime_cfg: dict,
        config_dir: Path,
    ):
        """
        Construct the preprocessing pipeline for inference.

        Args:
            preprocessing_cfg: Preprocessing configuration dictionary.
            model_cfg: Model configuration dictionary.
            runtime_cfg: Runtime-specific settings (Lambda vs local).
            config_dir: Base configuration directory.

        Returns:
            Initialized PreprocessingPipeline instance.
        """
        logger.info("Initializing preprocessing pipeline")

        return PreprocessingPipeline(
            config_paths={
                "preprocessing": preprocessing_cfg,
                "model": model_cfg,
            },
            raw_json_pattern=runtime_cfg["raw_json_pattern"],
            use_s3=runtime_cfg["use_s3"],
            s3_bucket=S3_BUCKET,
            s3_prefix=S3_PREFIX,
            model_config_path=config_dir / "model_config.yaml",
            model_name=model_cfg.get(
                "model_name",
                "xgboost_early_stopping_optuna_feature_eng_geoloc_exp",
            ),
            load_cache=runtime_cfg["load_pipeline_cache"],
            save_cache=runtime_cfg["save_pipeline_cache"],
        )

    def _load_inference_metadata(self, config_dir: Path):
        """
        Load inference metadata required to align preprocessing
        with training-time feature expectations.

        Args:
            config_dir: Base configuration directory.
        """
        logger.info("Loading inference metadata")

        inference_meta_path = config_dir / "inference_meta.pkl"
        if not inference_meta_path.exists():
            raise RuntimeError(
                "inference_meta.pkl not found. "
                "Generate it during training preprocessing."
            )

        with open(inference_meta_path, "rb") as f:
            inference_meta = pickle.load(f)

        self.pipeline.meta = inference_meta["meta"]
        self.pipeline.expected_columns = inference_meta["expected_columns"]

        logger.info(
            "Inference metadata loaded (%d expected columns)",
            len(self.pipeline.expected_columns),
        )

    def _load_geo_metadata(self, config_dir: Path):
        """
        Load geolocation and amenities metadata used
        during feature engineering.

        Args:
            config_dir: Base configuration directory.
        """
        logger.info("Loading geo and amenities metadata")

        geo_cache_file, amenities_df, amenity_radius_map = load_geo_config(
            config_dir / "model_config.yaml"
        )

        geo_cache_path = config_dir / geo_cache_file
        if not geo_cache_path.exists():
            raise RuntimeError(f"Geo cache file not found: {geo_cache_path}")

        lat_lon_cache = load_cache(geo_cache_path)

        self.pipeline.meta.update(
            {
                "geo_cache_file": geo_cache_file,
                "amenities_df": amenities_df,
                "amenity_radius_map": amenity_radius_map,
                "lat_lon_cache": lat_lon_cache,
                "use_amenities": amenities_df is not None,
                "use_geolocation": True,
            }
        )

        logger.info(
            "Geo loaded | amenities=%s | lat/lon cache size=%d",
            amenities_df.shape if amenities_df is not None else None,
            len(lat_lon_cache),
        )

    def _load_model(self, model_cfg: dict):
        logger.info("Loading ML model")
        self.model = load_production_model(model_cfg)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def initialize(self, config_dir: str):
        """
        Initialize preprocessing pipeline and ML model for inference.
        Safe to call multiple times (singleton).
        """
        if self._initialized:
            logger.info(
                "PipelineManager already initialized; " \
                "skipping re-initialization"
            )
            return self

        config_dir = Path(config_dir)

        preprocessing_cfg, model_cfg = self._load_configs(config_dir)
        runtime_cfg = self._detect_runtime()

        self.pipeline = self._build_preprocessing_pipeline(
            preprocessing_cfg=preprocessing_cfg,
            model_cfg=model_cfg,
            runtime_cfg=runtime_cfg,
            config_dir=config_dir,
        )

        self._load_inference_metadata(config_dir)
        self._load_geo_metadata(config_dir)

        self.model = self._load_model(model_cfg)

        self._initialized = True
        logger.info("Pipeline and model initialized successfully")
        return self

    # -------------------------------------------------------------------------
    # Scraping
    # -------------------------------------------------------------------------
    def scrape(self, url: str, headless: bool = True) -> Dict[str, Any]:
        """
        Scrape a Funda listing.

        Returns a consistent dict:
        {
            "success": bool,
            "url": str,
            "data": dict or None,
            "error": str or None
        }
        """
        result = scrape_listing(url, headless)
        # Ensure keys always exist
        return {
            "success": result.get("success", False),
            "url": url,
            "data": result.get("data"),
            "error": result.get("error"),
        }

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    def _extract_features_from_listing(
        self,
        listing: Dict[str, Any],
        drop_target: bool,
    ) -> Dict[str, Any]:
        """
        Run preprocessing pipeline on a single listing and
        return a flat feature dictionary.

        Args:
            listing: Raw listing dictionary.
            drop_target: Whether to drop target column.

        Returns:
            Dictionary of processed features.

        Raises:
            ValueError if preprocessing produces no output.
        """
        df_result = self.pipeline.preprocess_single(
            listing, drop_target=drop_target
        )

        if isinstance(df_result, pd.DataFrame):
            if df_result.empty:
                raise ValueError("Preprocessed DataFrame is empty.")
            return df_result.iloc[0].to_dict()

        return dict(df_result)

    def _log_preprocess_input(self, listing: Dict[str, Any]):
        """
        Log useful debug information about incoming listing structure.
        """
        logger.debug("Incoming preprocess request")
        logger.debug("Incoming listing keys: %s", list(listing.keys()))

        if "data" in listing and isinstance(listing["data"], dict):
            logger.debug(
                "Subkeys under 'data': %s", list(listing["data"].keys())
            )

    def _normalize_preprocess_input(
        self,
        listing: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """
        Detect whether listing already contains preprocessed features.

        Returns:
            Feature dictionary if already preprocessed, else None.
        """
        if "features" in listing:
            logger.info(
                "Listing already contains features; skipping preprocessing"
            )
            return listing["features"]

        return None

    def preprocess(
        self, listing: Dict[str, Any], drop_target: bool = True
    ) -> Dict[str, Any]:
        """
        Preprocess a single listing dict using the fitted pipeline.

        Returns:
        {
            "success": bool,
            "features": dict or None,
            "error": str or None
        }
        """
        if not self._initialized:
            raise RuntimeError("PipelineManager not initialized.")

        try:
            self._log_preprocess_input(listing)

            # Case 1: already preprocessed
            features = self._normalize_preprocess_input(listing)
            if features is None:
                features = self._extract_features_from_listing(
                    listing=listing,
                    drop_target=drop_target,
                )

            return {
                "success": True,
                "features": features,
                "error": None,
            }

        except Exception as e:
            logger.error("Error during preprocessing", exc_info=True)
            return {
                "success": False,
                "features": None,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def _to_feature_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert input features into a single-row DataFrame.

        Args:
            features: Feature dictionary or DataFrame-like object.

        Returns:
            Single-row pandas DataFrame.
        """
        if isinstance(features, dict):
            return pd.DataFrame([features])

        return pd.DataFrame(features)

    def _get_model_feature_names(self) -> list[str]:
        """
        Extract feature names expected by the trained model.

        Returns:
            List of feature names.

        Raises:
            ValueError if feature names cannot be determined.
        """
        if hasattr(self.model, "feature_names"):
            return list(self.model.feature_names)

        if hasattr(self.model, "get_attr"):
            feature_names = self.model.get_attr("feature_names")
            if feature_names:
                return list(feature_names)

        raise ValueError("Cannot extract feature names from model.")

    def _align_features_to_model(
        self,
        features_df: pd.DataFrame,
        model_features: list[str],
    ) -> pd.DataFrame:
        """
        Align input features to model's expected feature order.

        Logs missing and extra features.

        Args:
            features_df: Input feature DataFrame.
            model_features: Feature names expected by the model.

        Returns:
            Aligned DataFrame.
        """
        input_features = features_df.columns.tolist()

        missing = [f for f in model_features if f not in input_features]
        extra = [f for f in input_features if f not in model_features]

        if missing:
            logger.warning("Missing features in input: %s", missing)
        if extra:
            logger.warning("Extra features in input: %s", extra)

        return features_df.reindex(columns=model_features, fill_value=0)

    def _sanitize_numeric_features(
        self, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Drop non-numeric columns before prediction.

        Args:
            features_df: Feature DataFrame.

        Returns:
            Numeric-only DataFrame.
        """
        non_numeric = features_df.select_dtypes(
            exclude=["number", "bool"]
        ).columns

        if len(non_numeric) > 0:
            logger.info(
                "Dropping non-numeric columns before prediction: %s",
                list(non_numeric),
            )
            features_df = features_df.drop(columns=non_numeric)

        return features_df

    def _run_model_prediction(self, features_df: pd.DataFrame) -> float:
        """
        Run model prediction and undo log-transform if applicable.

        Args:
            features_df: Aligned numeric feature DataFrame.

        Returns:
            Predicted price value.
        """
        dmatrix = xgb.DMatrix(features_df)
        raw_pred = float(self.model.predict(dmatrix)[0])

        # Undo log-transform
        return np.expm1(raw_pred)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict price from preprocessed features.

        Returns:
        {
            "success": bool,
            "prediction": float or None,
            "features": dict or None,
            "error": str or None
        }
        """
        if not self._initialized:
            raise RuntimeError("PipelineManager not initialized.")

        try:
            features_df = self._to_feature_dataframe(features)

            model_features = self._get_model_feature_names()
            features_df = self._align_features_to_model(
                features_df, model_features
            )
            features_df = self._sanitize_numeric_features(features_df)

            prediction = self._run_model_prediction(features_df)

            return {
                "success": True,
                "prediction": prediction,
                "features": features_df.to_dict(orient="records")[0],
                "error": None,
            }

        except Exception as e:
            logger.error("Prediction failed", exc_info=True)
            return {
                "success": False,
                "prediction": None,
                "features": None,
                "error": str(e),
            }

    def convert_data_from_manual_input(self, user_input: dict) -> pd.DataFrame:
        """
        Convert manual user input into a full feature DataFrame
        aligned with training columns.
        """
        df = pd.DataFrame([user_input])

        # Fill missing expected columns
        for col in self.pipeline.expected_columns:
            if col not in df.columns:
                df[col] = 0 if col.startswith("has_") else 0.0

        # Compute log features
        for log_col, base_col in [
            ("log_size_num", "size_num"),
            ("log_num_facilities", "num_facilities"),
        ]:
            if log_col not in df.columns and base_col in df.columns:
                df[log_col] = np.log1p(df[base_col].fillna(0))

        # Final column order match
        df = df.reindex(columns=self.pipeline.expected_columns, fill_value=0)
        return df

    # -------------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------------
    def run_full_pipeline(
        self, url: str = None, manual_input: dict = None, headless: bool = True
    ) -> Dict[str, Any]:
        """
        Full end-to-end: scrape → preprocess → predict OR
        manual input → predict.

        Args:
            url: Funda URL to scrape.
            manual_input: dict with user-provided features.
            headless: headless mode for scraping.

        Returns:
            dict with prediction and features.
        """
        if not self._initialized:
            raise RuntimeError("PipelineManager not initialized.")

        # --- Decide data source ---
        if url:
            scrape_result = self.scrape(url, headless=headless)
            if not scrape_result["success"]:
                return scrape_result
            listing = scrape_result["data"]
            preprocess_result = self.preprocess(listing, drop_target=True)
            if not preprocess_result["success"]:
                return preprocess_result
            features = preprocess_result["features"]

        elif manual_input:
            listing = build_listing_from_manual_input(manual_input)

            preprocess_result = self.preprocess(listing, drop_target=True)
            if not preprocess_result["success"]:
                return preprocess_result

            features = preprocess_result["features"]

        else:
            return {
                "success": False,
                "prediction": None,
                "features": None,
                "error": "Either 'url' or 'manual_input' must be provided.",
            }

        # --- Predict ---
        prediction_result = self.predict(features)
        return prediction_result
