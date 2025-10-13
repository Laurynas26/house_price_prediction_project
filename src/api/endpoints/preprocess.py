from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.api.core.manager import PipelineManager

router = APIRouter()
manager = PipelineManager()  # shared singleton-style instance

@router.post("/")
def preprocess_single_listing(
    listing: Dict[str, Any],
    drop_target: bool = True
):
    """
    Preprocess a single listing using the fitted pipeline.
    Returns the feature vector aligned with the model training features.

    Args:
        listing (dict): Raw listing data (scraped or user-provided)
        drop_target (bool): Whether to remove target columns like price

    Returns:
        dict: {
            "success": bool,
            "features": dict of preprocessed features,
            "error": str or None
        }
    """
    try:
        # Let the manager handle all pipeline interaction
        features_df = manager.preprocess(listing, drop_target=drop_target)
        features_dict = features_df.to_dict(orient="records")[0]

        return {
            "success": True,
            "features": features_dict,
            "error": None
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")
