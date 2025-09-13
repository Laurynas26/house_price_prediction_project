import yaml


def load_model_config_and_search_space(config_path, model_name):
    """
    Load model params, fit params, and optional Optuna search space
    from a unified YAML.

    Returns:
        model_params: dict of fixed model parameters
        fit_params: dict of fixed fit parameters
        search_space: dict for Optuna hyperparameter tuning
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
