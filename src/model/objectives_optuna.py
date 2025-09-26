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
from src.model.cv_helpers import prepare_base_data
from src.features.feature_engineering import feature_engineering_cv as fe_cv
from src.features.feature_engineering.encoding import encode_train_val_only

prepare_fold_features = fe_cv.prepare_fold_features


def unified_objective(
    trial,
    model_name: str,
    df,
    features_config: str,
    model_config: str,
    use_log: bool = False,
    use_extended_features: bool = True,
    n_splits: int = 3,
) -> float:
    """
    Objective function for Optuna hyperparameter optimization with leakage-safe
    K-Fold cross-validation and fold-wise feature engineering.

    This function supports both baseline models (using minimal features with
    categorical encoding of `energy_label`) and full models with extended
    feature engineering.

    Workflow:
        1. Load model configuration and suggest hyperparameters via Optuna trial.
        2. Prepare base dataset (features + target).
        3. Apply optional log-transform on the target variable.
        4. Perform K-Fold CV with fold-wise data preparation:
            - Baseline: encode only categorical `energy_label`.
            - Extended: run full feature engineering per fold.
        5. Train and evaluate model on each fold using `ModelEvaluator`.
        6. Aggregate validation RMSE across folds and return the mean.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object for hyperparameter sampling.
    model_name : str
        Name of the model to train (e.g., "xgb", "rf", "linear").
    df : pandas.DataFrame
        Full input dataframe containing features and target.
    features_config : str
        Path or identifier for the feature configuration to use.
    model_config : str
        Path or identifier for the model configuration and search space.
    use_log : bool, default=False
        If True, apply log1p transform to the target during training and
        expm1 inverse-transform during evaluation.
    use_extended_features : bool, default=True
        If True, perform fold-wise extended feature engineering.
        If False, only encode `energy_label` and basic preprocessing.
    n_splits : int, default=5
        Number of folds for cross-validation.

    Returns
    -------
    float
        Mean validation RMSE across folds.

    Raises
    ------
    ValueError
        If the provided `model_name` is not supported.
    """

    # 1️⃣ Load model config and suggest trial parameters
    model_params, fit_params, search_space = (
        load_model_config_and_search_space(model_config, model_name)
    )
    model_params, fit_params = suggest_params_from_space(
        trial, model_params, fit_params, search_space
    )

    # 2️⃣ Base data prep
    X_full, y_full = prepare_base_data(df, features_config, model_name)

    # # 🔍 Debug: Print global columns
    # print("\n===== Feature Columns (X) =====")
    # print(list(X_full.columns))
    # print(f"\nNumber of features: {len(X_full.columns)}")

    # print("\n===== Target Column (y) =====")
    # print(y_full.name)
    # print(f"Number of target values: {len(y_full)}\n")

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

        # # 🔍 Debug: after raw split
        # print(f"\n--- Fold {fold} (raw split) ---")
        # print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)
        # print("y_train size:", len(y_train), "y_val size:", len(y_val))
        # print("All feature names:", list(X_train.columns))

        # Baseline: fold-wise encode only energy_label
        if not use_extended_features:
            if "energy_label" in X_train.columns:
                X_train, X_val, _ = encode_train_val_only(X_train, X_val)

            # Convert any remaining object columns to numeric safely
            for col in X_train.columns:
                if X_train[col].dtype == "object":
                    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                    X_val[col] = pd.to_numeric(X_val[col], errors="coerce")

            # Fill NaNs
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)

            # # 🔍 Debug: after baseline encoding
            # print(f"[Fold {fold}] After baseline encoding:")
            # print("X_train dtypes summary:\n", X_train.dtypes.value_counts())
            # print("Any NaNs left?", X_train.isna().sum().sum())

        # Full features: fold-wise feature engineering
        else:
            X_train, X_val, meta, fold_encoders = prepare_fold_features(
                X_train, X_val, use_extended_features=True
            )

        #     # 🔍 Debug: after extended feature engineering
        #     print(f"[Fold {fold}] After extended feature engineering:")
        #     print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)
        #     print("New features added:", meta.get("expanded_features", []))

        # # 🔍 Debug: right before training
        # print(f"[Fold {fold}] Final training matrix has {X_train.shape[1]} features")

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
        elif (
            "rf" in model_name.lower() or "random_forest" in model_name.lower()
        ):
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
