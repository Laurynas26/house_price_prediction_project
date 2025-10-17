from fastapi import APIRouter, HTTPException, Query
from src.api.core.manager import PipelineManager

router = APIRouter()
manager = PipelineManager()


@router.post("/")
def run_full_pipeline(
    url: str = Query(..., description="Funda listing URL to scrape and predict"),
    headless: bool = True,
):
    """
    Run the full pipeline in explicit steps:
    1. Scrape the Funda listing
    2. Preprocess the scraped listing
    3. Predict price using aligned features
    """
    if not manager._initialized:
        raise HTTPException(status_code=500, detail="PipelineManager not initialized")

    try:
        # 1️⃣ Scrape
        scrape_result = manager.scrape(url, headless=headless)
        if not scrape_result["success"]:
            raise HTTPException(status_code=400, detail=scrape_result["error"])

        listing_data = scrape_result["data"]

        # 2️⃣ Preprocess
        preprocess_result = manager.preprocess(listing_data, drop_target=True)
        if not preprocess_result["success"]:
            raise HTTPException(status_code=400, detail=preprocess_result["error"])

        features = preprocess_result["features"]

        # 3️⃣ Predict
        prediction_result = manager.predict(listing_data)  # Predict expects raw listing internally
        if not prediction_result["success"]:
            raise HTTPException(status_code=400, detail=prediction_result["error"])

        return {
            "url": url,
            "features": features,
            "prediction": prediction_result["prediction"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full pipeline failed: {e}")
