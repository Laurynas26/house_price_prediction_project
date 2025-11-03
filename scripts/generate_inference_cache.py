from pathlib import Path
import sys
import yaml
import pandas as pd

# -------------------- Project setup --------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.preprocessing_pipeline import PreprocessingPipeline
from src.api.core.mlflow_utils import load_latest_mlflow_model

# -------------------- Paths and configs --------------------
ROOT = Path(__file__).resolve().parents[1]

PREPROCESSING_CFG_PATH = ROOT / "config/preprocessing_config.yaml"
MODEL_CFG_PATH = ROOT / "config/model_config.yaml"

with open(PREPROCESSING_CFG_PATH) as f:
    preprocessing_cfg = yaml.safe_load(f)

with open(MODEL_CFG_PATH) as f:
    model_cfg = yaml.safe_load(f)

config_paths = {
    "preprocessing": preprocessing_cfg.get("preprocessing", {}),
    "model": model_cfg.get("model", {}),
}

# -------------------- Pipeline settings --------------------
RAW_JSON_PATTERN = ROOT / "data/parsed_json/*.json"

raw_files = list((ROOT / "data/parsed_json").glob("*.json"))
if not raw_files:
    raise FileNotFoundError(
        f"No raw JSON files found in {ROOT / 'data/parsed_json'}"
    )

MODEL_SCOPE = "xgboost_early_stopping_optuna_feature_eng_geoloc_exp"
CACHE_DIR = ROOT / "data/cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# -------------------- Instantiate pipeline --------------------
pipeline = PreprocessingPipeline(
    config_paths=config_paths,
    raw_json_pattern=str(RAW_JSON_PATTERN),
    load_cache=False,
    save_cache=True,
    cache_dir=str(CACHE_DIR),
    model_config_path=MODEL_CFG_PATH,
    model_name=MODEL_SCOPE,
)

# -------------------- Run pipeline for meta --------------------
pipeline.run(smart_cache=False)

# Inject geo/amenities info into meta
geo_cfg = preprocessing_cfg.get("geo_feature_exp", {})
pipeline.meta["amenities_df"] = pd.read_csv(
    ROOT / geo_cfg.get("amenities_file")
)
pipeline.meta["amenity_radius_map"] = geo_cfg.get("amenity_radius_map")
pipeline.meta["geo_cache_file"] = str(ROOT / geo_cfg.get("geo_cache_file"))

# -------------------- Load MLflow model --------------------
model = load_latest_mlflow_model("XGB_Optuna_LogTransformed_feature_eng_best")

# -------------------- Update expected_columns from model --------------------
expected_columns = model.feature_names
print(
    f"✅ Expected columns from MLflow model: {len(expected_columns)} features"
)

# -------------------- Save updated inference cache --------------------
inference_meta = {"meta": pipeline.meta, "expected_columns": expected_columns}
pipeline.cache.save(
    inference_meta, "inference_meta", scope=pipeline.model_name
)
print(
    f"✅ Inference cache updated at {CACHE_DIR} with MLflow model feature names"
)