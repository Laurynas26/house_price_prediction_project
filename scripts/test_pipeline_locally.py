from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.api.core.manager import PipelineManager

CONFIG_DIR = Path(__file__).parents[1] / "config"

# --- Example manual input for testing ---
# manual_input = {
#     "size": 200,
#     "num_facilities": 3,
#     "energy_label": "D",
#     "roof_type": "flat",
#     "ownership_type": "owner",
#     "neighborhood": "centrum",
#     "has_garden": 1,
#     "has_balcony": 1,
#     "has_sauna": 1,
# }
manual_input = {
    # --- Core ---
    "price": None,
    "size": 100,
    "contribution_vve": 250,
    "external_storage": 6,
    "year_of_construction": 1998,
    "nr_rooms": 4,
    "bathrooms": 1,
    "toilets": 2,
    "bedrooms": 3,
    # --- Facilities (string or list both OK) ---
    "facilities": [
        "sauna",
        "lift",
        "mechanische ventilatie",
        "zonnepanelen",
    ],
    # --- Outdoor ---
    "outdoor_features": {
        "garden": True,
        "balcony": True,
        "roof_terrace": False,
    },
    # --- Geo ---
    "postal_code": "1017AB",
    "address": "Keizersgracht 123",
    "neighborhood_details": {
        "name": "Centrum",
    },
    # --- Meta / categorical ---
    "roof_type": "flat",
    "status": "Verkocht",
    "ownership_type": "owner",
    "location": "Residential",
    "energy_label": "D",
    "located_on": "ground_floor",
    "backyard": "yes",
    "balcony": "yes",
    # --- Usually empty but schema-safe ---
    "cadastral_parcels": [],
    "ownership_situations": [],
    "charges": [],
}


# --- Initialize the pipeline manager ---
manager = PipelineManager()
manager.initialize(CONFIG_DIR)

# --- Run full pipeline for manual input ---
result = manager.run_full_pipeline(manual_input=manual_input)

print("Prediction result:")
print(result)
