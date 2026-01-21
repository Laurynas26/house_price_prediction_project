"""
DEPRECATED ENDPOINTS — DO NOT USE IN EC2 SERVICE

Kept for reference only.
Preprocessing + prediction now live in AWS Lambda.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.api.core.manager import PipelineManager

router = APIRouter()
manager = PipelineManager()  # shared singleton-style instance


@router.post("/")
def preprocess_single_listing(listing: Dict[str, Any], drop_target: bool = True):
    """
    Preprocess a single listing using the fitted pipeline.
    Accepts either:
    - raw listing dict
    - or scrape response: {"url": ..., "success": True, "data": {...},
    "error": None}
    """
    try:
        # --- Handle wrapped scrape response ---
        if "data" in listing and isinstance(listing["data"], dict):
            listing_to_process = listing["data"]
        else:
            listing_to_process = listing

        # --- Optional diagnostic logging ---
        print("\n=== Incoming /preprocess request ===")
        print(f"Listing keys: {list(listing_to_process.keys())}")

        # --- Run pipeline preprocessing ---
        result = manager.preprocess(listing_to_process, drop_target=drop_target)

        # --- Validation: manager-level error ---
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error"))

        print("✅ Preprocessing completed successfully.")
        return result

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        print("\n--- ERROR DURING PREPROCESSING ---")
        print(f"Type: {type(e)}")
        print(f"Message: {e}")
        traceback.print_exc(limit=8)
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")
