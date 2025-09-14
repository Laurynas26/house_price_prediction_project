import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from src.model.utils import load_model_config_and_search_space, suggest_params_from_space
from src.model.evaluate import ModelEvaluator
from src.features.data_prep_for_modelling.data_preparation import prepare_data
from src.features.feature_engineering.encoding import encode_train_val_only
from src.features.feature_engineering.feature_engineering import prepare_features_train_val


def unified_objective(
    trial,
    model_name: str,
    df,
    features_config: str,
    model_config: str,
    use_log: bool = False,
    use_extended_features: bool = False,
    n_splits: int = 5,
):
    """
    Optuna objective function with K-Fold cross-validation and fold-wise feature engineering.

    This function:
        - Loads model hyperparameters from config and suggests Optuna trial values.
        - Prepares the dataset without encoding the 'energy_label' globally to prevent leakage.
        - Performs K-Fold CV.
        - Applies fold-wise extended feature engineering (if requested).
        - Encodes 'energy_label' safely per fold.
        - Trains and evaluates the model per fold.
        - Returns the average validation RMSE for minimization in Optuna.

    Args:
        trial: Optuna trial object for hyperparameter suggestions.
        model_name (str): Name of the model to train ('xgboost', 'random_forest', 'linear', etc.).
        df (pd.DataFrame): Input dataframe containing features and target.
        features_config (str): Path to YAML feature configuration.
        model_config (str): Path to YAML model configuration.
        use_log (bool): Whether to apply log-transform on target.
        use_extended_features (bool): If True, apply full fold-wise feature engineering.
        n_splits (int): Number of K-Fold splits.

    Returns:
        float: Average validation RMSE across all folds (minimization target for Optuna).
    """

    # 1️⃣ Load model config and suggest trial parameters
    model_params, fit_params, search_space = load_model_config_and_search_space(model_config, model_name)
    model_params, fit_params = suggest_params_from_space(trial, model_params, fit_params, search_space)

    # 2️⃣ Prepare dataset without encoding 'energy_label' globally
    X_full, _, y_full, _, _, _, _, feature_encoders = prepare_data(
        df,
        config_path=features_config,
        model_name=model_name,
        use_extended_features=False,
        cv=True  # ensures energy_label is not globally encoded
    )

    # Optional target log-transform
    target_transform = np.log1p if use_log else None
    inverse_transform = np.expm1 if use_log else None

    # 3️⃣ Setup K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_rmse_list = []

    for train_idx, val_idx in kf.split(X_full):
        X_train, X_val = X_full.iloc[train_idx].copy(), X_full.iloc[val_idx].copy()
        y_train, y_val = y_full.iloc[train_idx].copy(), y_full.iloc[val_idx].copy()

        # 4️⃣ Fold-wise feature engineering (skip energy_label)
        if use_extended_features:
            X_train, X_val, _, fe_encoders = prepare_features_train_val(
                X_train, X_val, encode_energy=False
            )
        else:
            fe_encoders = {}

        # 5️⃣ Encode energy_label safely per fold
        X_train, X_val, energy_encoder = encode_train_val_only(X_train, X_val)
        fe_encoders["energy_label"] = energy_encoder

        # 6️⃣ Initialize evaluator
        evaluator = ModelEvaluator(target_transform=target_transform, inverse_transform=inverse_transform)

        # 7️⃣ Train and evaluate per fold
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
        elif "rf" in model_name.lower() or "random_forest" in model_name.lower():
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

    # 8️⃣ Return average validation RMSE
    return float(np.mean(val_rmse_list))
