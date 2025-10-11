from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any
import yaml

from src.data_loading.data_loading.data_loader import load_data_from_json
from src.data_loading.preprocessing.preprocessing import preprocess_df
from src.data_loading.preprocessing.imputation import impute_missing_values
from src.features.data_prep_for_modelling.data_preparation import (
    prepare_data_from_config,
)

from src.features.feature_engineering.feature_engineering import (
    prepare_features_test,
)
from src.data_loading.data_cache import CacheManager


@dataclass
class PipelineResult:
    """Container for final processed datasets and metadata."""

    X_train: pd.DataFrame
    X_val: Optional[pd.DataFrame]
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: Optional[pd.Series]
    y_test: pd.Series
    scaler: Optional[Any]
    meta: Dict
    df_clean: pd.DataFrame


class PreprocessingPipeline:
    """
    Modular preprocessing pipeline for real estate data.

    This class orchestrates data loading, preprocessing, imputation, and
    feature engineering steps. It includes optional caching via CacheManager
    to avoid recomputation when configurations or raw data remain unchanged.
    """

    def __init__(
        self,
        config_paths: Dict[str, Any],
        raw_json_pattern: str,
        load_cache: bool = True,
        save_cache: bool = False,
        cache_dir: str = "data/cache",
        model_config_path: Optional[Path] = None,
        model_name: str = None,
    ):
        """
        Args:
            config_paths: Dictionary of loaded configuration dicts
                          (keys: 'preprocessing', 'model', etc.)
            raw_json_pattern: Glob pattern for JSON input data
            cache_enabled: Whether to enable caching between runs
            cache_dir: Directory where cache files are stored
        """
        self.config_paths = config_paths
        self.raw_json_pattern = raw_json_pattern
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.cache = CacheManager(cache_dir=cache_dir)
        self.model_config_path = model_config_path
        self.model_name = model_name

        # Storage
        self.df_raw = None
        self.df_clean = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.scaler = None
        self.meta = {}

        # Define sequential steps
        self.steps = [
            self.load_data,
            self.preprocess,
            self.impute,
            self.drop_missing_target,
            self.feature_engineering,
        ]

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------
    def run(self, smart_cache: bool = True) -> PipelineResult:
        """
        Run the full preprocessing pipeline.

        Args:
            smart_cache: If True, skips steps that already have cached results.
        """
        for step in self.steps:
            step_name = step.__name__

            # Check if we can skip this step entirely
            cache_key_map = {
                "load_data": ("df_raw", None),
                "preprocess": (
                    "df_preprocessed",
                    self.config_paths.get("preprocessing"),
                ),
                "feature_engineering": (
                    "feature_eng_result",
                    self.config_paths.get("model"),
                    "xgboost_early_stopping_optuna_feature_eng_geoloc_exp",
                ),
            }

            if smart_cache and step_name in cache_key_map:
                key_info = cache_key_map[step_name]
                cache_key = key_info[0]
                cfg = key_info[1]
                scope = key_info[2] if len(key_info) > 2 else None

                if self.cache.exists(cache_key, cfg, scope=scope):
                    print(
                        f"[SMART CACHE] Skipping {step_name}, loading cached result"
                    )
                    cached_data = self.cache.load(cache_key, cfg, scope=scope)

                    # Inject cached data directly
                    if step_name == "load_data":
                        self.df_raw = cached_data
                    elif step_name == "preprocess":
                        self.df_clean = cached_data
                    elif step_name == "feature_engineering":
                        (
                            self.X_train,
                            self.X_val,
                            self.X_test,
                            self.y_train,
                            self.y_val,
                            self.y_test,
                            self.scaler,
                            self.meta,
                        ) = cached_data
                    continue  # skip step entirely

            # Run the step if no cache
            step()

        return PipelineResult(
            X_train=self.X_train,
            X_val=self.X_val,
            X_test=self.X_test,
            y_train=self.y_train,
            y_val=self.y_val,
            y_test=self.y_test,
            scaler=self.scaler,
            meta=self.meta,
            df_clean=self.df_clean,
        )

    # -------------------------------------------------------------------------
    # Step Implementations
    # -------------------------------------------------------------------------
    def load_data(self):
        """Load raw data from JSON, optionally using cache."""
        if self.load_cache and self.cache.exists("df_raw", config=None):
            self.df_raw = self.cache.load("df_raw")
        else:
            self.df_raw = load_data_from_json(self.raw_json_pattern)
            print(f"Loaded {len(self.df_raw)} raw listings")
            if self.save_cache:
                self.cache.save(self.df_raw, "df_raw")

    def preprocess(self):
        cfg = self.config_paths["preprocessing"]
        if self.load_cache and self.cache.exists(
            "df_preprocessed", cfg, scope="preprocessing"
        ):
            self.df_clean = self.cache.load(
                "df_preprocessed", cfg, scope="preprocessing"
            )
        else:
            self.df_clean = preprocess_df(
                self.df_raw,
                drop_raw=cfg.get("drop_raw", True),
                numeric_cols=cfg.get("numeric_cols", []),
            )
            print(f"Preprocessed data: {self.df_clean.shape}")
            if self.save_cache:
                self.cache.save(
                    self.df_clean,
                    "df_preprocessed",
                    cfg,
                    scope="preprocessing",
                )

    def impute(self):
        """Impute missing values using specified strategy."""
        imputation_cfg = self.config_paths["preprocessing"].get(
            "imputation", {}
        )
        self.df_clean = impute_missing_values(self.df_clean, imputation_cfg)
        print("Imputation done")

    def drop_missing_target(self):
        """Drop rows with missing target variable and redundant columns."""
        self.df_clean = self.df_clean[self.df_clean["price_num"].notna()]
        if "living_area" in self.df_clean.columns:
            self.df_clean.drop(columns=["living_area"], inplace=True)
        print("Dropped missing target rows")

    def feature_engineering(self):
        """Prepare train/test/validation sets based on model configuration."""

        model_name = self.model_name

        # For caching: get the model config dict (from config_paths)
        model_cfg_dict = self.config_paths.get("model", {})

        # Check cache first
        if self.load_cache and self.cache.exists(
            "feature_eng_result", model_cfg_dict, scope=model_name
        ):
            data = self.cache.load(
                "feature_eng_result", model_cfg_dict, scope=model_name
            )
            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                self.scaler,
                self.meta,
            ) = data
            print(
                f"Loaded cached feature-engineered data: {self.X_train.shape}"
            )
            return

        # --- Run feature engineering ---
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.X_val,
            self.y_val,
            self.scaler,
            self.meta,
        ) = prepare_data_from_config(
            df=self.df_clean,
            config_path=self.model_config_path,  # Pass the Path, not dict
            model_name=model_name,
            enable_cache_save=True,
        )
        print(f"Feature engineering done: Train shape {self.X_train.shape}")

        # --- Save to cache ---
        if self.save_cache:
            self.cache.save(
                (
                    self.X_train,
                    self.X_val,
                    self.X_test,
                    self.y_train,
                    self.y_val,
                    self.y_test,
                    self.scaler,
                    self.meta,
                ),
                "feature_eng_result",
                model_cfg_dict,
                scope=model_name,
            )

    def preprocess_single(
        self, listing: dict, drop_target: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess a single listing dict using the fitted pipeline.
        Returns a scaled DataFrame aligned with X_train columns.

        Handles missing or extra columns gracefully.
        """
        if self.df_clean is None or self.meta is None or self.scaler is None:
            raise RuntimeError("Pipeline must be run first to fit transforms!")

        # Convert dict to DataFrame
        df = pd.DataFrame([listing])

        # -------------------------------
        # Preprocessing
        # -------------------------------
        cfg = self.config_paths["preprocessing"]
        df = preprocess_df(
            df,
            drop_raw=cfg.get("drop_raw", True),
            numeric_cols=cfg.get("numeric_cols", []),
        )

        # Imputation
        imputation_cfg = cfg.get("imputation", {})
        df = impute_missing_values(df, imputation_cfg)

        # -------------------------------
        # Feature engineering (test mode)
        # -------------------------------
        df = prepare_features_test(
            df,
            self.meta,
            use_geolocation=bool(self.meta.get("geo_meta")),
            use_amenities=bool(self.meta.get("amenity_meta")),
            amenities_df=self.meta.get("amenities_df"),
            amenity_radius_map=self.meta.get("amenity_radius_map"),
            geo_cache_file=self.meta.get("geo_cache_file"),
        )

        # Drop address
        if "address" in df.columns:
            df.drop(columns="address", inplace=True)

        # -------------------------------
        # Align columns with training set
        # -------------------------------
        X_train_cols = self.X_train.columns

        # Add missing columns with zeros
        for col in X_train_cols:
            if col not in df.columns:
                df[col] = 0

        # Remove extra columns that are not in training
        extra_cols = [c for c in df.columns if c not in X_train_cols]
        if extra_cols:
            df.drop(columns=extra_cols, inplace=True)

        # Reorder columns to match X_train
        df = df[X_train_cols]

        # -------------------------------
        # Apply scaler
        # -------------------------------
        df_scaled = pd.DataFrame(
            self.scaler.transform(df), columns=df.columns, index=df.index
        )
        if drop_target:
            df_scaled = df_scaled.drop(
                columns=["price", "price_num"], errors="ignore"
            )

        return df_scaled
