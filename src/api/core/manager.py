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
                "PipelineManager already initialized; "
                "skipping re-initialization"
            )
            return self

        config_dir = Path(config_dir)

        # ------------------------------------------------------------------
        # Load configs
        # ------------------------------------------------------------------
        with open(config_dir / "preprocessing_config.yaml") as f:
            preprocessing_cfg = yaml.safe_load(f)

        with open(config_dir / "model_config.yaml") as f:
            model_cfg = yaml.safe_load(f)

        # ------------------------------------------------------------------
        # Detect environment
        # ------------------------------------------------------------------
        running_on_lambda = "AWS_LAMBDA_FUNCTION_NAME" in os.environ

        if running_on_lambda:
            logger.info("Running on AWS Lambda")
            use_s3 = True
            raw_json_pattern = None
            load_pipeline_cache = True
            save_pipeline_cache = False
        else:
            logger.info("Running locally: cache-only inference mode")
            use_s3 = False
            raw_json_pattern = None
            load_pipeline_cache = False
            save_pipeline_cache = False

        # ------------------------------------------------------------------
        # Create preprocessing pipeline (NO execution here)
        # ------------------------------------------------------------------
        self.pipeline = PreprocessingPipeline(
            config_paths={
                "preprocessing": preprocessing_cfg,
                "model": model_cfg,
            },
            raw_json_pattern=raw_json_pattern,
            use_s3=use_s3,
            s3_bucket=S3_BUCKET,
            s3_prefix=S3_PREFIX,
            model_config_path=config_dir / "model_config.yaml",
            model_name=model_cfg.get(
                "model_name",
                "xgboost_early_stopping_optuna_feature_eng_geoloc_exp",
            ),
            load_cache=load_pipeline_cache,
            save_cache=save_pipeline_cache,
        )

        # ------------------------------------------------------------------
        # Load inference_meta (UNHASHED, UN-SCOPED)
        # ------------------------------------------------------------------

        logger.info("Loading inference metadata from disk")

        # inference_meta_path = Path("data/cache/inference_meta.pkl")
        inference_meta_path = config_dir / "inference_meta.pkl"

        if not inference_meta_path.exists():
            raise RuntimeError(
                "inference_meta.pkl not found at\n "
                "data/cache/inference_meta.pkl\n"
                "Generate it once during training preprocessing."
            )

        with open(inference_meta_path, "rb") as f:
            inference_meta = pickle.load(f)

        self.pipeline.meta = inference_meta["meta"]
        self.pipeline.expected_columns = inference_meta["expected_columns"]

        logger.info(
            "Loaded inference metadata (%d expected columns)",
            len(self.pipeline.expected_columns),
        )

        # ------------------------------------------------------------------
        # Load geo & amenities metadata (same as training)
        # ------------------------------------------------------------------

        geo_cache_file, amenities_df, amenity_radius_map = load_geo_config(
            config_dir / "model_config.yaml"
        )

        # --- Resolve geo cache path relative to config_dir ---
        geo_cache_path = config_dir / geo_cache_file

        if not geo_cache_path.exists():
            raise RuntimeError(
                f"Geo cache file not found at {geo_cache_path}. "
                "Ensure it is bundled in config/ for Lambda."
            )

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
            "Geo metadata loaded | amenities shape=%s | lat/lon cache size=%d",
            amenities_df.shape if amenities_df is not None else None,
            len(lat_lon_cache),
        )

        # ------------------------------------------------------------------
        # Load ML model from MLflow
        # ------------------------------------------------------------------

        self.model = load_production_model(model_cfg)

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
    def preprocess(
        self, listing: Dict[str, Any], drop_target: bool = True
    ) -> Dict[str, Any]:
        """
        Preprocess a single listing dict using the fitted pipeline.
        Returns a dict with structure:
        {
            "success": bool,
            "features": dict,
            "error": str or None
        }
        """
        if not self._initialized:
            raise RuntimeError("PipelineManager not initialized.")

        try:
            logger.debug("Incoming preprocess request")
            logger.debug("Incoming listing keys: %s", list(listing.keys()))

            if "data" in listing:
                logger.debug(
                    "Subkeys under 'data': %s", list(listing["data"].keys())
                )

            # If already preprocessed dict, skip reprocessing
            if "features" in listing:
                features_dict = listing["features"]
            else:
                df_result = self.pipeline.preprocess_single(
                    listing, drop_target=drop_target
                )

                # Safely convert to dict
                if isinstance(df_result, pd.DataFrame):
                    if not df_result.empty:
                        features_dict = df_result.iloc[0].to_dict()
                    else:
                        raise ValueError("Preprocessed DataFrame is empty.")
                else:
                    features_dict = dict(df_result)

            return {"success": True, "features": features_dict, "error": None}

        except Exception as e:
            logger.error(
                "Error during preprocessing",
                exc_info=True,
            )
            return {"success": False, "features": None, "error": str(e)}

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict price from preprocessed features
        (dict aligned with training columns).

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
            # Convert dict -> DataFrame
            if isinstance(features, dict):
                features_df = pd.DataFrame([features])
            else:
                features_df = pd.DataFrame(features)

            # --- Get model feature names ---
            model_features = (
                self.model.feature_names
                if hasattr(self.model, "feature_names")
                else (
                    self.model.get_attr("feature_names")
                    if hasattr(self.model, "get_attr")
                    else None
                )
            )

            if model_features is None:
                raise ValueError("Cannot extract feature names from model.")

            # --- Align columns ---
            df_features = features_df.columns.tolist()
            missing = [f for f in model_features if f not in df_features]
            extra = [f for f in df_features if f not in model_features]

            if missing:
                logger.warning("Missing features in input: %s", missing)
            if extra:
                logger.warning("Extra features in input: %s", extra)

            features_df = features_df.reindex(
                columns=model_features, fill_value=0
            )

            # --- Drop any non-numeric columns ---
            non_numeric = features_df.select_dtypes(
                exclude=["number", "bool"]
            ).columns
            if len(non_numeric) > 0:
                logger.info(
                    "Dropping non-numeric columns before prediction: %s",
                    list(non_numeric),
                )

                features_df = features_df.drop(columns=non_numeric)

            # --- Run prediction ---
            dmatrix = xgb.DMatrix(features_df)
            raw_pred = float(self.model.predict(dmatrix)[0])

            # Undo log-transform if applicable
            pred_value = np.expm1(raw_pred)

            return {
                "success": True,
                "prediction": pred_value,
                "features": features_df.to_dict(orient="records")[0],
                "error": None,
            }

        except Exception as e:
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
