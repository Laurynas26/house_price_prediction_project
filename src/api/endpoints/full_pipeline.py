"""
DEPRECATED ENDPOINTS — DO NOT USE IN EC2 SERVICE

Kept for reference only.
Preprocessing + prediction now live in AWS Lambda.
"""

from fastapi import APIRouter, HTTPException, Query
from src.api.core.manager import PipelineManager

router = APIRouter()
manager = PipelineManager()


@router.post("/")
def full_pipeline(
    url: str = Query(..., description="Funda listing URL"),
    headless: bool = True,
):
    """
    Run the full pipeline:
    1. Scrape the listing
    2. Preprocess the scraped data
    3. Predict price from features
    Returns a dict with success, features, and prediction.
    """
    if not manager._initialized:
        raise HTTPException(
            status_code=500, detail="PipelineManager not initialized"
        )

    try:
        # --- Scrape ---
        scrape_result = manager.scrape(url, headless=headless)
        if not scrape_result.get("success", False):
            return {
                "success": False,
                "error": f"Scrape failed: {scrape_result.get('error')}",
            }

        listing_data = scrape_result.get("data")
        if listing_data is None:
            return {"success": False, "error": "Scrape returned no data"}

        # --- Preprocess ---
        preprocess_result = manager.preprocess(listing_data, drop_target=True)
        if not preprocess_result.get("success", False):
            return {
                "success": False,
                "error": f"Preprocess failed: "
                f"{preprocess_result.get('error')}",
            }

        features = preprocess_result.get("features")
        if features is None:
            return {
                "success": False,
                "error": "Preprocess returned no features",
            }

        # --- Predict ---
        prediction_result = manager.predict(features)
        if not prediction_result.get("success", False):
            return {
                "success": False,
                "error": f"Prediction failed: "
                f"{prediction_result.get('error')}",
            }

        return {
            "success": True,
            "url": url,
            "features": features,
            "prediction": prediction_result.get("prediction"),
        }

    except Exception as e:
        return {"success": False, "error": f"Full pipeline failed: {e}"}
