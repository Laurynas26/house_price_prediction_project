from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.api.core.manager import PipelineManager

router = APIRouter()
manager = PipelineManager()  # shared singleton-style instance


@router.post("/")
def preprocess_single_listing(
    listing: Dict[str, Any], drop_target: bool = True
):
    """
    Preprocess a single listing using the fitted pipeline.
    Returns the feature vector aligned with the model training features.
    """
    try:
        result = manager.preprocess(listing, drop_target=drop_target)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return result  # already {"success": True, "features": {...}, "error": None}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Preprocessing failed: {e}"
        )
