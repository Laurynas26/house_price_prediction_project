import json
from pathlib import Path
import pandas as pd
import yaml

from src.api.core.manager import PipelineManager

# --- Cold-start singleton: keep manager in memory ---
manager = PipelineManager()
ROOT = Path(__file__).resolve().parents[2]

# Initialize on cold start
with open(ROOT / "config/preprocessing_config.yaml") as f:
    preprocessing_cfg = yaml.safe_load(f)
geo_cfg = preprocessing_cfg.get("geo_feature_exp", {})

manager.initialize(config_dir=str(ROOT / "config"))
manager.pipeline.meta["amenities_df"] = pd.read_csv(
    ROOT / geo_cfg.get("amenities_file")
)
manager.pipeline.meta["amenity_radius_map"] = geo_cfg.get("amenity_radius_map")
manager.pipeline.meta["geo_cache_file"] = str(ROOT / geo_cfg.get("geo_cache_file"))

print("[Lambda] PipelineManager initialized")


def lambda_handler(event, context):
    """
    Lambda entry point for API Gateway
    Expects JSON payload like:
    {
        "url": "https://www.funda.nl/detail/...",
        "headless": true
    }
    or
    {
        "manual_input": {...}
    }
    """
    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)
        if not body:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing request body"}),
            }

        url = body.get("url")
        manual_input = body.get("manual_input")
        headless = body.get("headless", True)

        # Run the pipeline
        if url or manual_input:
            result = manager.run_full_pipeline(url=url, manual_input=manual_input, headless=headless)
            return {"statusCode": 200, "body": json.dumps(result)}
        else:
            return {"statusCode": 400, "body": json.dumps({"error": "Provide either 'url' or 'manual_input'"})}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
