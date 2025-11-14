from pathlib import Path
from typing import Dict, Any
import yaml
import xgboost as xgb
import pandas as pd
import numpy as np
import os

from src.scraper.core import scrape_listing
from src.features.preprocessing_pipeline import PreprocessingPipeline
from src.api.core.mlflow_utils import load_latest_mlflow_model

RAW_JSON_PATTERN = Path(__file__).parents[3] / "data/parsed_json/*.json"


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
        Load configs, run preprocessing pipeline, and load model.
        Can be safely called multiple times (singleton).

        Args:
            config_dir: Directory with preprocessing.yml and model.yml
            model_path: Path to trained model (.joblib/.pkl)
        """
        if self._initialized:
            print("[Manager] Already initialized, skipping re-initialization.")
            return self

        config_dir = Path(config_dir)

        # Load configs
        with open(config_dir / "preprocessing_config.yaml") as f:
            preprocessing_cfg = yaml.safe_load(f)
        with open(config_dir / "model_config.yaml") as f:
            model_cfg = yaml.safe_load(f)

        # ------------------------------
        # Detect environment
        # ------------------------------
        running_on_lambda = "LAMBDA_TASK_ROOT" in os.environ

        if running_on_lambda:
            print("[Manager] Running on AWS Lambda: using S3 storage")
            use_s3 = True
            local_raw_pattern = None  # Lambda has no CSV/JSON
            load_cache = True  # Always load from S3
            save_cache = False  # Do not write cache in Lambda
        else:
            print("[Manager] Running locally: using local JSON files")
            use_s3 = False
            local_raw_pattern = str(RAW_JSON_PATTERN)
            load_cache = True
            save_cache = True

        # ------------------------------
        # Create preprocessing pipeline
        # ------------------------------
        self.pipeline = PreprocessingPipeline(
            config_paths={
                "preprocessing": preprocessing_cfg,
                "model": model_cfg,
            },
            raw_json_pattern=local_raw_pattern,
            use_s3=use_s3,
            model_config_path=config_dir / "model_config.yaml",
            model_name=model_cfg.get(
                "model_name",
                "xgboost_early_stopping_optuna_feature_eng_geoloc_exp",
            ),
            load_cache=load_cache,
            save_cache=save_cache,
        )

        self.pipeline.run(smart_cache=True)
        print("[DEBUG] After pipeline run:")
        print(
            " - X_train:",
            type(self.pipeline.X_train),
            getattr(self.pipeline.X_train, "shape", None),
        )
        print(" - scaler:", type(self.pipeline.scaler))
        print(
            " - meta keys:",
            list(self.pipeline.meta.keys()) if self.pipeline.meta else None,
        )

        # --- Safety fallback ---
        if self.pipeline.meta is None or self.pipeline.X_train is None:
            print("[Manager] ⚠️ Cached pipeline incomplete, refitting...")
            self.pipeline.run(smart_cache=False)

        # --- MLflow model loading ---
        production_model_name = model_cfg.get("production_model_name")
        if not production_model_name:
            raise RuntimeError(
                "Model name must be specified in model_config.yaml"
            )

        experiment_name = "house_price_prediction"
        try:
            self.model = load_latest_mlflow_model(
                production_model_name, experiment_name=experiment_name
            )
        except RuntimeError as e:
            print("[Manager] MLflow error:", e)
            print(
                "Available MLflow folders:",
                list((Path(__file__).parents[3] / "logs/mlruns").glob("*")),
            )
            raise

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
            # Step 1: transform input
            df_manual = self.convert_data_from_manual_input(manual_input)

            # Step 2: full preprocessing
            preprocess_result = self.preprocess(
                df_manual.iloc[0].to_dict(), drop_target=True
            )
            if not preprocess_result["success"]:
                return preprocess_result

            # Step 3: predict
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
