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


def unified_objective(
    trial,
    model_name: str,
    df,
    features_config: str,
    model_config: str,
    use_log: bool = False,
    use_extended_features: bool = True,
    n_splits: int = 5,
) -> float:
    """
    Optuna objective function with K-Fold CV and leakage-safe fold-wise feature engineering.

    This function:
      - Loads model hyperparameters from config and suggests Optuna trial values.
      - Prepares the dataset using `prepare_base_data` (no global encoding).
      - Performs K-Fold cross-validation.
      - Applies fold-wise feature engineering and safe energy_label encoding using `prepare_fold_features`.
      - Trains and evaluates the model per fold.
      - Returns the average validation RMSE for Optuna minimization.

    Args:
        trial: Optuna trial object for hyperparameter suggestions.
        model_name: Model identifier ('xgboost', 'random_forest', 'linear', etc.).
        df: Input dataframe containing features and target.
        features_config: Path to YAML feature configuration (used inside helpers).
        model_config: Path to YAML model configuration (used inside helpers).
        use_log: If True, applies log-transform to the target.
        use_extended_features: If True, applies full fold-wise feature engineering.
        n_splits: Number of K-Fold splits.

    Returns:
        float: Average validation RMSE across all folds.
    """

    # 1️⃣ Load model config and suggest trial parameters
    model_params, fit_params, search_space = (
        load_model_config_and_search_space(model_config, model_name)
    )
    model_params, fit_params = suggest_params_from_space(
        trial, model_params, fit_params, search_space
    )

    # Base data prep (before fold-wise FE)
    X_full, y_full = prepare_base_data(df, features_config, model_name)


    # Optional log-transform on target
    target_transform = np.log1p if use_log else None
    inverse_transform = np.expm1 if use_log else None

    # 3️⃣ Setup K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_rmse_list = []

    for train_idx, val_idx in kf.split(X_full):
        X_train, X_val = (
            X_full.iloc[train_idx].copy(),
            X_full.iloc[val_idx].copy(),
        )
        y_train, y_val = (
            y_full.iloc[train_idx].copy(),
            y_full.iloc[val_idx].copy(),
        )

        # 4️⃣ Fold-wise feature engineering + safe energy_label encoding
        X_train, X_val, meta, fold_encoders = prepare_fold_features(
            X_train, X_val, use_extended_features=use_extended_features
        )

        # 5️⃣ Initialize evaluator
        evaluator = ModelEvaluator(
            target_transform=target_transform,
            inverse_transform=inverse_transform,
        )

        # 6️⃣ Train & evaluate model
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

        val_rmse_list.append(results["test_rmse"])

    # 7️⃣ Return mean validation RMSE
    return float(np.mean(val_rmse_list))
