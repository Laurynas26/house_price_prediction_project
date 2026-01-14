from pathlib import Path
from typing import Dict, Any
import yaml
import xgboost as xgb
import pandas as pd
import numpy as np
import os

from src.scraper.core import scrape_listing
from src.features.preprocessing_pipeline import (
    PreprocessingPipeline,
)
from src.api.core.mlflow_utils import load_latest_mlflow_model
from src.features.data_prep_for_modelling.data_preparation import (
    load_geo_config,
)
from src.features.feature_engineering.location_feature_enrichment import (
    load_cache,
)

EXPECTED_SCHEMA = {
    "price": (None, (int, float, str, type(None))),
    "contribution_vve": (None, (int, float, str, type(None))),
    "size": (None, (int, float, str, type(None))),
    "external_storage": (None, (int, float, str, type(None))),
    "year_of_construction": (None, (int, float, str, type(None))),
    "nr_rooms": (None, (int, float, str, type(None))),
    "bathrooms": (None, (int, float, str, type(None))),
    "toilets": (None, (int, float, str, type(None))),
    "bedrooms": (None, (int, float, str, type(None))),
    "facilities": ("", (str, list, type(None))),
    "outdoor_features": ({}, (dict, type(None))),
    "cadastral_parcels": ([], (list, type(None))),
    "ownership_situations": ([], (list, type(None))),
    "charges": ([], (list, type(None))),
    "postal_code": (None, (str, type(None))),
    "neighborhood_details": ({}, (dict, type(None))),
    "address": (None, (str, type(None))),
    "roof_type": (None, (str, type(None))),
    "status": (None, (str, type(None))),
    "ownership_type": (None, (str, type(None))),
    "location": (None, (str, type(None))),
    "energy_label": (None, (str, type(None))),
    "located_on": (None, (str, type(None))),
    "backyard": (None, (str, type(None))),
    "balcony": (None, (str, type(None))),
}


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
            print("[Manager] Already initialized, skipping re-initialization.")
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
            print("[Manager] Running on AWS Lambda")
            use_s3 = True
            raw_json_pattern = None
            load_pipeline_cache = True
            save_pipeline_cache = False
        else:
            print("[Manager] Running locally: cache-only inference mode")
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
        print("[Manager] Loading inference_meta directly")

        try:
            inference_meta = self.pipeline.cache.load("inference_meta")
        except FileNotFoundError:
            raise RuntimeError(
                "inference_meta.pkl not found.\n"
                "You must generate it once during training preprocessing."
            )

        self.pipeline.meta = inference_meta["meta"]
        self.pipeline.expected_columns = inference_meta["expected_columns"]

        print(
            f"[Manager] Loaded inference_meta | "
            f"{len(self.pipeline.expected_columns)} expected columns"
        )

        # ------------------------------------------------------------------
        # Load geo & amenities metadata (same as training)
        # ------------------------------------------------------------------
        from src.data_loading.geo_utils import load_cache, load_geo_config

        geo_cache_file, amenities_df, amenity_radius_map = load_geo_config(
            config_dir / "model_config.yaml"
        )

        lat_lon_cache = load_cache(geo_cache_file)

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

        print(
            f"[Manager] Geo loaded | "
            f"Amenities: {amenities_df.shape if amenities_df is not None else None}, "
            f"Lat/Lon cache size: {len(lat_lon_cache)}"
        )

        # ------------------------------------------------------------------
        # Load ML model from MLflow
        # ------------------------------------------------------------------
        production_model_name = model_cfg.get("production_model_name")
        if not production_model_name:
            raise RuntimeError("production_model_name missing in model_config.yaml")

        experiment_name = "house_price_prediction"

        self.model = load_latest_mlflow_model(
            production_model_name,
            experiment_name=experiment_name,
        )

        self._initialized = True
        print("[Manager] Pipeline and model initialized successfully.")
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
            print("=== Incoming /preprocess request ===")
            print("Incoming listing keys:", list(listing.keys()))
            if "data" in listing:
                print("Subkeys under 'data':", list(listing["data"].keys()))

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
            import traceback

            print("\n--- ERROR INSIDE manager.preprocess ---")
            print(f"Type: {type(e)}")
            print(f"Message: {e}")
            traceback.print_exc(limit=10)
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

            if missing or extra:
                print(f"[DEBUG] Missing in input: {missing}")
                print(f"[DEBUG] Extra in input: {extra}")

            features_df = features_df.reindex(
                columns=model_features, fill_value=0
            )

            # --- Drop any non-numeric columns ---
            non_numeric = features_df.select_dtypes(
                exclude=["number", "bool"]
            ).columns
            if len(non_numeric) > 0:
                print(
                    f"[INFO] Dropping non-numeric columns before "
                    f"prediction: {list(non_numeric)}"
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
            listing = self.manual_input_to_listing(manual_input)

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

    # def manual_input_to_listing(self, manual_input: dict) -> dict:
    #     return {
    #         "price": None,
    #         "living_area": manual_input.get("size_num"),
    #         "energy_label": manual_input.get("energy_label"),
    #         "roof_type": manual_input.get("roof_type"),
    #         "ownership_type": manual_input.get("ownership_type"),
    #         "neighborhood": manual_input.get("neighborhood"),
    #         "facilities": {
    #             "garden": manual_input.get("has_garden", 0),
    #             "balcony": manual_input.get("has_balcony", 0),
    #         },
    #     }
    def manual_input_to_listing(self, manual_input: dict) -> dict:
        """
        Build a schema-safe listing dict from manual input.
        """
        listing = {}

        for field, (default, _) in EXPECTED_SCHEMA.items():
            listing[field] = manual_input.get(field, default)

        return listing
