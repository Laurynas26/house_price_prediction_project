import yaml

def load_model_config(config_path, model_name):
    """Load model- and fit-related params from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg[model_name]

    model_params = model_cfg.get("model_params", {})
    fit_params = model_cfg.get("fit_params", {})

    return model_params, fit_params