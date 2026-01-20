import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.model.utils import (
    load_model_config_and_search_space,
    suggest_params_from_space,
)
from src.model.evaluate import ModelEvaluator
from src.model.cv_helpers import prepare_base_data, prepare_fold_features
from src.features.feature_engineering.encoding import encode_train_val_only


def unified_objective(
    trial,
    model_name: str,
    df,
    features_config: str,
    model_config: str,
    use_log: bool = False,
    use_extended_features: bool = True,
    use_geo_amenities: bool = True,  # 👈 toggle added here
    n_splits: int = 3,
    enable_cache_save: bool = False,
) -> float:
    """
    Objective function for Optuna hyperparameter optimization with leakage-safe
    K-Fold cross-validation and fold-wise feature engineering.

    Parameters
    ----------
    use_extended_features : bool
        If True, perform extended fold-wise feature engineering.
    use_geo_amenities : bool
        If True, load geo/amenities from YAML and include them in
        feature expansion.
    """

    # 1️⃣ Load model config and suggest trial parameters
    model_params, fit_params, search_space = load_model_config_and_search_space(
        model_config, model_name
    )
    model_params, fit_params = suggest_params_from_space(
        trial, model_params, fit_params, search_space
    )

    # 2️⃣ Base data prep
    X_full, y_full = prepare_base_data(
        df, features_config, model_name, extended_fe=use_extended_features
    )

    # Optional log-transform on target
    target_transform = np.log1p if use_log else None
    inverse_transform = np.expm1 if use_log else None

    # 3️⃣ K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_rmse_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full), 1):
        X_train, X_val = (
            X_full.iloc[train_idx].copy(),
            X_full.iloc[val_idx].copy(),
        )
        y_train, y_val = (
            y_full.iloc[train_idx].copy(),
            y_full.iloc[val_idx].copy(),
        )

        # Baseline: no extended FE
        if not use_extended_features:
            if "energy_label" in X_train.columns:
                X_train, X_val, _ = encode_train_val_only(X_train, X_val)

            for col in X_train.columns:
                if X_train[col].dtype == "object":
                    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                    X_val[col] = pd.to_numeric(X_val[col], errors="coerce")

            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)

        # Full features: fold-wise FE (with optional geo/amenities)
        else:
            X_train, X_val, meta, fold_encoders = (
                prepare_fold_features(
                    X_train,
                    X_val,
                    features_config=features_config,  # 👈 YAML-based configs
                    use_extended_features=True,
                    enable_cache_save=enable_cache_save,
                )
                if use_geo_amenities
                else prepare_fold_features(
                    X_train,
                    X_val,
                    features_config=None,  # 👈 no geo config loaded
                    use_extended_features=True,
                    enable_cache_save=enable_cache_save,
                )
            )

        # 4️⃣ Initialize evaluator
        evaluator = ModelEvaluator(
            target_transform=target_transform,
            inverse_transform=inverse_transform,
        )

        # 5️⃣ Train & evaluate
        if "xgb" in model_name.lower():
            _, _, _, _, results = evaluator.evaluate(
                model=None,
                X_train=X_train,
                y_train=y_train,
                X_test=X_val,
                y_test=y_val,
                X_val=X_val,
                y_val=y_val,
                model_params=model_params,
                fit_params=fit_params,
                use_xgb_train=True,
            )
        elif "rf" in model_name.lower():
            model = RandomForestRegressor(**model_params)
            _, _, _, _, results = evaluator.evaluate(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_val,
                y_test=y_val,
                X_val=X_val,
                y_val=y_val,
                fit_params=fit_params,
            )
        elif "linear" in model_name.lower():
            model = LinearRegression(**model_params)
            _, _, _, _, results = evaluator.evaluate(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_val,
                y_test=y_val,
                X_val=X_val,
                y_val=y_val,
                fit_params=fit_params,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        val_rmse_list.append(results["val_rmse"])

    return float(np.mean(val_rmse_list))
