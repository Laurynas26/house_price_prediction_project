from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.api.core.manager import PipelineManager

CONFIG_DIR = Path(__file__).parents[1] / "config"

# --- Example manual input for testing ---
manual_input = {
    "size_num": 100,
    "num_facilities": 3,
    "energy_label": "D",
    "roof_type": "flat",
    "ownership_type": "owner",
    "neighborhood": "centrum",
    "has_garden": 1,
    "has_balcony": 1,
    "has_sauna": 1,
}

# --- Initialize the pipeline manager ---
manager = PipelineManager()
manager.initialize(CONFIG_DIR)

# --- Run full pipeline for manual input ---
result = manager.run_full_pipeline(manual_input=manual_input)

print("Prediction result:")
print(result)
