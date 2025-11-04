from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from src.data_loading.data_loading.data_loader import (
    load_data_from_json,
    json_to_df_raw_strict,
)
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
    """Full preprocessing and feature engineering pipeline
    with caching support."""

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
        self.config_paths = config_paths
        self.raw_json_pattern = raw_json_pattern
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.cache = CacheManager(cache_dir=cache_dir)
        self.model_config_path = model_config_path
        self.model_name = model_name

        # Data storage
        self.df_raw = None
        self.df_clean = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.scaler = None
        self.meta = {}
        self.expected_columns = []

        # Pipeline steps
        self.steps = [
            self.load_data,
            self.preprocess,
            self.impute,
            self.drop_missing_target,
            self.feature_engineering,
        ]

        # Cache key map for DRY smart caching
        self.cache_key_map = {
            "load_data": ("df_raw", None, None),
            "preprocess": (
                "df_preprocessed",
                self.config_paths,
                "preprocessing",
            ),
            "feature_engineering": (
                "feature_eng_result",
                self.config_paths,
                self.model_name,
            ),
        }

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------
    def run(self, smart_cache: bool = True) -> PipelineResult:
        for step in self.steps:
            step_name = step.__name__

            key_info = self.cache_key_map.get(step_name)
            if smart_cache and key_info:
                cache_key, cfg, scope = key_info
                if self.cache.exists(cache_key, cfg, scope=scope):
                    print(
                        f"[SMART CACHE] Skipping {step_name}, "
                        f"loading cached result"
                    )
                    cached_data = self.cache.load(cache_key, cfg, scope=scope)
                    self._restore_from_cache(step_name, cached_data)
                    continue

            step()

        # Capture expected feature schema
        if self.X_train is not None:
            self.expected_columns = self.X_train.columns.tolist()

        # Save minimal inference schema
        if self.save_cache and self.X_train is not None and self.meta:
            inference_meta = {
                "meta": self.meta,
                "expected_columns": self.X_train.columns.tolist(),
            }
            self.cache.save(
                inference_meta, "inference_meta", scope=self.model_name
            )
            print(
                "[CACHE] Saved inference_meta for future single listing"
                " preprocessing."
            )

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

    def _restore_from_cache(self, step_name: str, cached_data: Any):
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

    # -------------------------------------------------------------------------
    # Step Implementations
    # -------------------------------------------------------------------------
    def load_data(self):
        if self.load_cache and self.cache.exists("df_raw", config=None):
            self.df_raw = self.cache.load("df_raw")
        else:
            self.df_raw = load_data_from_json(self.raw_json_pattern)
            print(f"Loaded {len(self.df_raw)} raw listings")
            if self.save_cache:
                self.cache.save(self.df_raw, "df_raw")

    def preprocess(self):
        cfg = self.config_paths.get("preprocessing", {})
        if self.load_cache and self.cache.exists(
            "df_preprocessed", self.config_paths, scope="preprocessing"
        ):
            self.df_clean = self.cache.load(
                "df_preprocessed", self.config_paths, scope="preprocessing"
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
                    self.config_paths,
                    scope="preprocessing",
                )

    def impute(self):
        cfg = self.config_paths.get("preprocessing", {})
        imputation_cfg = cfg.get("imputation", {})
        self.df_clean = impute_missing_values(self.df_clean, imputation_cfg)
        print("Imputation done")

    def drop_missing_target(self):
        self.df_clean = self.df_clean[self.df_clean["price_num"].notna()]
        print("Dropped missing target rows")

    def feature_engineering(self):
        if self.load_cache and self.cache.exists(
            "feature_eng_result", self.config_paths, scope=self.model_name
        ):
            data = self.cache.load(
                "feature_eng_result", self.config_paths, scope=self.model_name
            )
            self._restore_from_cache("feature_engineering", data)
            print(
                f"Loaded cached feature-engineered data: {self.X_train.shape}"
            )
            return

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
            config_path=self.model_config_path,
            model_name=self.model_name,
            enable_cache_save=True,
        )
        print(f"Feature engineering done: Train shape {self.X_train.shape}")

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
                self.config_paths,
                scope=self.model_name,
            )

    # -------------------------------------------------------------------------
    # Single Listing Preprocessing (XGBoost-compatible)
    # -------------------------------------------------------------------------
    def preprocess_single(
        self, listing: dict, drop_target: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess a single listing JSON dict into a model-ready
        feature DataFrame.
        Ensures schema consistency with training data.
        """
        if (
            not self.meta or getattr(self, "X_train", None) is None
        ) and self.cache.exists("inference_meta", scope=self.model_name):
            print("[INFO] Loading inference metadata from cache...")
            inference_meta = self.cache.load(
                "inference_meta", scope=self.model_name
            )
            self.meta = inference_meta["meta"]
            self.expected_columns = inference_meta["expected_columns"]
        elif not self.meta or getattr(self, "X_train", None) is None:
            raise RuntimeError(
                "Pipeline must be run first or inference cache missing."
            )

        df_raw = json_to_df_raw_strict(listing)
        cfg = self.config_paths.get("preprocessing", {})
        df = preprocess_df(
            df_raw,
            drop_raw=cfg.get("drop_raw", True),
            numeric_cols=cfg.get("numeric_cols", []),
        )
        df = impute_missing_values(df, cfg.get("imputation", {}))
        df = prepare_features_test(
            df,
            meta=self.meta,
            use_geolocation=bool(self.meta.get("geo_meta")),
            use_amenities=bool(self.meta.get("amenity_meta")),
            amenities_df=self.meta.get("amenities_df"),
            amenity_radius_map=self.meta.get("amenity_radius_map"),
            geo_cache_file=self.meta.get("geo_cache_file"),
        )
        df = ensure_all_categorical_columns(df, self.meta)

        # Align with training columns
        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = 0 if col.startswith("has_") else pd.NA
        df = df.reindex(columns=self.expected_columns)

        if drop_target:
            df = df.drop(columns=["price", "price_num"], errors="ignore")

        # --- Drop features that never existed in training ---
        extra_cols = [c for c in df.columns if c not in self.expected_columns]
        if extra_cols:
            print(
                f"[INFO] Dropping {len(extra_cols)} extra cols "
                f"not seen in training: {extra_cols[:10]}..."
            )
            df.drop(columns=extra_cols, inplace=True, errors="ignore")

        # --- Compute missing log features if they are not present ---
        for log_col, base_col in [
            ("log_size_num", "size_num"),
            ("log_num_facilities", "num_facilities"),
        ]:
            if log_col not in df.columns and base_col in df.columns:
                df[log_col] = np.log1p(df[base_col].fillna(0).astype(float))

        # --- Ensure all training columns exist ---
        missing_cols = [
            c for c in self.expected_columns if c not in df.columns
        ]
        for col in missing_cols:
            df[col] = 0 if col.startswith("has_") else pd.NA

        # --- Final reindex (strict column order match) ---
        df = df.reindex(columns=self.expected_columns, fill_value=0)

        # Drop target if required
        if drop_target:
            df = df.drop(columns=["price", "price_num"], errors="ignore")

        # --- Drop postal_code_clean if it survived ---
        if "postal_code_clean" in df.columns:
            df.drop(columns=["postal_code_clean"], inplace=True)


        return df


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def ensure_facility_columns(
    df: pd.DataFrame, key_facilities: list[str]
) -> pd.DataFrame:
    """
    Ensure all facility indicator columns exist in df.
    Missing ones are added with 0; extra columns are ignored.
    """
    for col in key_facilities:
        if col not in df.columns:
            df[col] = 0
    return df[
        key_facilities + [c for c in df.columns if c not in key_facilities]
    ]


def ensure_all_categorical_columns(
    df: pd.DataFrame, meta: dict
) -> pd.DataFrame:
    """
    Ensures that all one-hot encoded categorical columns seen
    during training exist.
    Handles:
      - Facilities
      - Energy label
      - Roof type
      - Ownership type
      - Neighborhood
    Missing ones are added with 0.
    """

    if "key_facilities" in meta:
        df = ensure_facility_columns(df, meta["key_facilities"])
    for cat_key in [
        "energy_label_cols",
        "roof_type_cols",
        "ownership_type_cols",
        "neighborhood_cols",
    ]:
        for col in meta.get(cat_key, []):
            if col not in df.columns:
                df[col] = 0
    return df
