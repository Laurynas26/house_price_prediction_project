from pathlib import Path
import sys
import pprint
import pandas as pd
import yaml

# Ensure project path is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api.core.manager import PipelineManager

# -------------------------
# Initialize manager
# -------------------------
manager = PipelineManager().initialize(config_dir="config/")

# Inject geo/amenities info from preprocessing_config.yaml
ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "config/preprocessing_config.yaml") as f:
    preprocessing_cfg = yaml.safe_load(f)

geo_cfg = preprocessing_cfg.get("geo_feature_exp", {})
manager.pipeline.meta["amenities_df"] = pd.read_csv(
    ROOT / geo_cfg.get("amenities_file")
)
manager.pipeline.meta["amenity_radius_map"] = geo_cfg.get("amenity_radius_map")
manager.pipeline.meta["geo_cache_file"] = str(ROOT / geo_cfg.get("geo_cache_file"))

# -------------------------
# Option 1: Run full pipeline from a Funda URL
# -------------------------
url = "https://www.funda.nl/detail/koop/amsterdam/appartement-bilderdijkkade-75-b/43179082/"
result_url = manager.run_full_pipeline(url=url, headless=True)

print("\n=== Full pipeline result (scraped URL) ===")
pprint.pprint(result_url)
if result_url.get("success"):
    print(f"Predicted price: €{result_url['prediction']:.0f}")

# -------------------------
# Option 2: Run full pipeline from manual input
# -------------------------
manual_input = {
    "size_num": 66,
    "nr_rooms": 3,
    "bathrooms": 1,
    "toilets": 1,
    "external_storage_num": 4,
    "floor_level": 2,
    "balcony_flag": 1,
    "energy_label_encoded": 6,
    "num_facilities": 4,
    "luxury_score": 1,
    "has_glasvezelkabel": 1,
    "has_lift": 1,
    "has_mechanische_ventilatie": 1,
    "postal_code_clean": "1100 LK",
    "lat": 52.372,      # approximate coordinates for this postal code
    "lon": 4.895,
}


result_manual = manager.run_full_pipeline(manual_input=manual_input)

print("\n=== Full pipeline result (manual input) ===")
pprint.pprint(result_manual)
if result_manual.get("success"):
    print(f"Predicted price: €{result_manual['prediction']:.0f}")
