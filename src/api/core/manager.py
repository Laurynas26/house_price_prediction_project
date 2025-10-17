from pathlib import Path
from typing import Dict, Any
import yaml
import mlflow.xgboost
import xgboost as xgb
import pandas as pd

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

        # Initialize pipeline
        self.pipeline = PreprocessingPipeline(
            config_paths={
                "preprocessing": preprocessing_cfg,
                "model": model_cfg,
            },
            raw_json_pattern=str(RAW_JSON_PATTERN),
            model_config_path=config_dir / "model_config.yaml",
            model_name=model_cfg.get(
                "model_name",
                "xgboost_early_stopping_optuna_feature_eng_geoloc_exp",
            ),
            load_cache=True,
            save_cache=False,
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
            # If already preprocessed dict, skip reprocessing
            if "features" in listing:
                features_dict = listing["features"]
            else:
                result = self.pipeline.preprocess_single(
                    listing, drop_target=drop_target
                )
                features_dict = (
                    result.to_dict(orient="records")[0]
                    if hasattr(result, "to_dict")
                    else dict(result)
                )

            return {"success": True, "features": features_dict, "error": None}

        except Exception as e:
            return {"success": False, "features": None, "error": str(e)}

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict price from preprocessed features (dict aligned with training columns).
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

            dmatrix = xgb.DMatrix(features_df)
            pred_value = float(self.model.predict(dmatrix)[0])

            return {
                "success": True,
                "prediction": pred_value,
                "features": features,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "prediction": None,
                "features": None,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------------
    def run_full_pipeline(
        self, url: str, headless: bool = True
    ) -> Dict[str, Any]:
        """
        Full end-to-end: scrape → preprocess → predict.
        """
        if not self._initialized:
            raise RuntimeError("PipelineManager not initialized.")

        # Step 1: Scrape
        scrape_result = self.scrape(url, headless=headless)
        if not scrape_result["success"]:
            return scrape_result

        listing = scrape_result["data"]

        # Step 2: Preprocess
        preprocess_result = self.preprocess(listing, drop_target=True)
        if not preprocess_result["success"]:
            return preprocess_result

        features = preprocess_result["features"]

        # Step 3: Predict
        prediction_result = self.predict(features)

        return {**prediction_result, "url": url}
