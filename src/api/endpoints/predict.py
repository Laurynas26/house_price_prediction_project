from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.api.core.manager import PipelineManager

router = APIRouter()
manager = PipelineManager()


@router.post("/")
def predict_price(preprocessed_listing: Dict[str, Any]):
    """
    Predict house price for a listing that has already been preprocessed.

    Expects a dict like the `features` field from the preprocess endpoint:
    {"size_num": 86, "nr_rooms": 3, ...}
    """
    if not manager._initialized:
        raise HTTPException(
            status_code=500, detail="PipelineManager not initialized"
        )
    try:
        # Validate input
        if "features" in preprocessed_listing:
            features = preprocessed_listing["features"]
        else:
            features = preprocessed_listing  # assume direct features dict

        # --- Run prediction ---
        prediction_result = manager.predict(features)

        return prediction_result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
