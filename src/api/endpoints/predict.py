from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.api.core.manager import PipelineManager

router = APIRouter()
manager = PipelineManager()


@router.post("/")
def predict_price(listing: Dict[str, Any]):
    """
    Predict house price for a given listing.
    Expects a dict of property features (e.g., from scraper or manual input).
    """
    if not manager._initialized:
        raise HTTPException(
            status_code=500, detail="PipelineManager not initialized"
        )
    try:
        result = manager.predict(listing)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
