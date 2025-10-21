from pathlib import Path
import sys
import pprint
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api.core.manager import PipelineManager

# -------------------------
# Initialize manager
# -------------------------
# This will load preprocessing/model configs, run the pipeline and load the MLflow model.
manager = PipelineManager().initialize(config_dir="config/")

# -------------------------
# Option A: inline scraped JSON 
# -------------------------
scraped_json = {
  "url": "https://www.funda.nl/detail/koop/amsterdam/appartement-bilderdijkkade-75-b/43179082/",
  "success": True,
  "data": {
    "price": "€ 750.000 k.k.",
    "address": "Bilderdijkkade 75-B",
    "postal_code": "1053 VK Amsterdam",
    "neighborhood": "Da Costabuurt-Zuid",
    "status": "N/A",
    "size": "86 m²",
    "bedrooms": "2",
    "energy_label": "A",
    "neighborhood_details": {
      "Inhabitants in neighborhood": "2.235",
      "Families with children": "11%",
      "Price per m² in neighborhood": "€ 9.673"
    },
    "contribution": "N/A",
    "year_of_construction": "1997",
    "roof_type": "Plat dak bedekt met bitumineuze dakbedekking",
    "living_area": "86 m²",
    "external_storage": "4 m²",
    "balcony": "Balkon aanwezig",
    "nr_rooms": "3",
    "bathrooms": "1",
    "toilets": "1",
    "located_on": "2e woonlaag",
    "facilities": "Glasvezelkabel, lift, mechanische ventilatie, en TV kabel",
    "cadastral_parcels": [
      {
        "parcel": "AMSTERDAM Q 7862",
        "link": "/detail/koop/amsterdam/appartement-bilderdijkkade-75-b/43179082/kadaster"
      }
    ],
    "ownership_situations": [
      "Gemeentelijk eigendom belast met erfpacht (einddatum erfpacht: 01-09-2081)"
    ],
    "charges": [
      "Afgekocht tot 01-09-2081"
    ],
    "outdoor_features": {
      "Ligging": "Aan water en in woonwijk",
      "Tuin": None,
      "Achtertuin": None,
      "Ligging tuin": None
    }
  },
  "error": None
}

# -------------------------
# Option B: load scraped JSON from file (alternate)
# -------------------------
# Uncomment and edit path if you'd rather load from a file
# sample_file = Path(__file__).resolve().parents[1] / "scripts" / "sample_scraped_listing.json"
# scraped_json = json.loads(sample_file.read_text())

# -------------------------
# Use the 'data' sub-dict for preprocessing
# manager.preprocess expects either the inner listing dict (recommended) or full dict if you handle it.
# -------------------------
listing_data = scraped_json.get("data")
if listing_data is None:
    raise RuntimeError("No 'data' key in scraped JSON — provide the inner listing dictionary.")

print("\n=== Running single-listing preprocessing ===")
result = manager.preprocess(listing_data, drop_target=True)

print("\n=== Preprocess Result ===")
pprint.pprint(result)

if result.get("success"):
    features = result["features"]
    print("\n-- Feature vector keys (first 30) --")
    print(list(features.keys())[:30])
    print("\n-- Example feature values (first 10) --")
    for k in list(features.keys())[:10]:
        print(f"{k}: {features[k]}")
else:
    print("\nPreprocessing failed:", result.get("error"))
