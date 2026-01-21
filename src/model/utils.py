import yaml
import numpy as np


def load_model_config_and_search_space(config_path, model_name):
    """
    Load model parameters, fit parameters, and Optuna search space
    from a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML file.
    model_name : str
        Key corresponding to the desired model in YAML.

    Returns
    -------
    model_params : dict
        Fixed model hyperparameters.
    fit_params : dict
        Fixed fit parameters (e.g., early stopping, num rounds).
    search_space : dict
        Optuna search space dictionary.
    """

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg[model_name]

    model_params = model_cfg.get("model_params", {})
    fit_params = model_cfg.get("fit_params", {})
    search_space = model_cfg.get("search_space", {})

    return model_params, fit_params, search_space


def suggest_params_from_space(trial, model_params, fit_params, search_space):
    """
    Suggest hyperparameters for Optuna trial based on search_space.
    Returns updated model_params and fit_params dictionaries.
    """
    model_params = dict(model_params)
    fit_params = dict(fit_params)

    for name, spec in search_space.items():
        if spec["type"] == "int":
            val = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            val = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "categorical":
            val = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported param type: {spec['type']}")

        # Decide if the parameter belongs to model_params or fit_params
        if name in model_params or name not in fit_params:
            model_params[name] = val
        else:
            fit_params[name] = val

    return model_params, fit_params


def huber_loss(
    y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0
) -> float:
    """
    Huber loss metric.

    Args:
        y_true: true target values
        y_pred: predicted values
        delta: threshold for switching between squared and linear loss

    Returns:
        float: mean Huber loss
    """
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small, squared_loss, linear_loss))
