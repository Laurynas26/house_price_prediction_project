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
        # --- Optional diagnostic logging ---
        print("\n=== Incoming /preprocess request ===")
        print(f"Incoming listing keys: {list(listing.keys())}")

        # If nested, print subkeys for clarity
        if isinstance(listing.get("data"), dict):
            print(f"Subkeys under 'data': {list(listing['data'].keys())}")

        # --- Run pipeline preprocessing ---
        result = manager.preprocess(listing, drop_target=drop_target)

        # --- Validation: manager-level error ---
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error"))

        print("✅ Preprocessing completed successfully.")
        return result
    except HTTPException:

        raise

    except Exception as e:
        # --- Diagnostic error trace ---
        import traceback

        print("\n--- ERROR DURING PREPROCESSING ---")
        print(f"Type: {type(e)}")
        print(f"Message: {e}")
        traceback.print_exc(limit=8)

        raise HTTPException(
            status_code=500, detail=f"Preprocessing failed: {e}"
        )
