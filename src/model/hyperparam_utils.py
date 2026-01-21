import yaml


def load_search_space(config_path, model_name):
    """Load hyperparameter search space for a model from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg[model_name]
    base_params = {k: v for k, v in model_cfg.items() if k != "search_space"}
    search_space = model_cfg.get("search_space", {})
    return base_params, search_space


def suggest_params_from_space(trial, base_params, search_space):
    """Generate parameters from Optuna trial using search space definition."""
    params = dict(base_params)  # fixed params

    for name, spec in search_space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported param type: {spec['type']}")

    return params
