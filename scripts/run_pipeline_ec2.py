from pathlib import Path
import sys
import pprint
import pandas as pd
import yaml

# Add project path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api.core.manager import PipelineManager

# -------------------------
# S3 Configuration
# -------------------------
USE_S3 = True
S3_BUCKET = "my-bucket-name"      # <- replace with your bucket
S3_PREFIX = "parsed_json/"         # <- replace with your prefix
CONFIG_DIR = Path("/path/to/configs")  # preprocessing_config.yaml & model_config.yaml

# -------------------------
# Initialize PipelineManager
# -------------------------
manager = PipelineManager(
    use_s3=USE_S3,
    s3_bucket=S3_BUCKET,
    s3_prefix=S3_PREFIX,
    raw_json_pattern=None,  
    load_cache=True,
    save_cache=False
).initialize(config_dir=CONFIG_DIR)

# -------------------------
# Inject geo/amenities info into pipeline meta
# -------------------------
with open(CONFIG_DIR / "preprocessing_config.yaml") as f:
    preprocessing_cfg = yaml.safe_load(f)

geo_cfg = preprocessing_cfg.get("geo_feature_exp", {})

manager.pipeline.meta["amenities_df"] = pd.read_csv(
    CONFIG_DIR / geo_cfg.get("amenities_file")
)
manager.pipeline.meta["amenity_radius_map"] = geo_cfg.get("amenity_radius_map")
manager.pipeline.meta["geo_cache_file"] = str(CONFIG_DIR / geo_cfg.get("geo_cache_file"))

print("\n=== Pipeline initialized and S3 data loaded ===")
print(f"X_train shape: {manager.pipeline.X_train.shape}")
print(f"Keys in meta: {list(manager.pipeline.meta.keys())}")

# -------------------------
# Optional: Test single listing preprocessing + prediction
# -------------------------
# Example JSON structure (replace with real listing from S3 if desired)
sample_listing = {
    "price": "€ 750.000 k.k.",
    "address": "Bilderdijkkade 75-B",
    "postal_code": "1053 VK Amsterdam",
    "neighborhood": "Da Costabuurt-Zuid",
    "status": "N/A",
    "size": "86 m²",
    "bedrooms": "2",
    "energy_label": "A",
    "year_of_construction": "1997",
    "facilities": ["Glasvezelkabel", "lift", "mechanische ventilatie", "TV kabel"],
    "outdoor_features": {"Ligging": "Aan water en in woonwijk"},
    "cadastral_parcels": [{"parcel": "AMSTERDAM Q 7862"}],
    "ownership_situations": ["Gemeentelijk eigendom belast met erfpacht"],
    "charges": ["Afgekocht tot 01-09-2081"],
}

print("\n=== Preprocessing single listing ===")
preprocess_result = manager.preprocess(sample_listing, drop_target=True)
pprint.pprint(preprocess_result)

if preprocess_result.get("success"):
    features = preprocess_result["features"]

    print("\n=== Running prediction ===")
    prediction_result = manager.predict(features)
    pprint.pprint(prediction_result)

    if prediction_result.get("success"):
        print(f"\nPredicted price: €{prediction_result['prediction']:.0f}")
    else:
        print("Prediction failed:", prediction_result["error"])
else:
    print("Preprocessing failed:", preprocess_result.get("error"))
