from fastapi import APIRouter, HTTPException, Query
from src.api.core.manager import PipelineManager

router = APIRouter()
manager = PipelineManager()


@router.post("/")
def run_full_pipeline(
    url: str = Query(
        ..., description="Funda listing URL to scrape and predict"
    ),
    headless: bool = True,
):
    """
    Run the full pipeline:
    - Scrape the Funda listing
    - Preprocess the data
    - Predict price (if model is loaded)
    """
    if not manager._initialized:
        raise HTTPException(
            status_code=500, detail="PipelineManager not initialized"
        )
    try:
        return manager.run_full_pipeline(url, headless=headless)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution failed: {e}"
        )
