from fastapi import APIRouter, HTTPException, Query
from src.scraper.core import scrape_listing

router = APIRouter()

@router.post("/")
def scrape_and_return(
    url: str = Query(..., description="Funda listing URL"), 
    headless: bool = True
):
    """
    Scrape a Funda listing and return parsed structured data with debug info.
    """
    try:
        result = scrape_listing(url, headless=headless)
        return {
            "url": url,
            "success": result["success"],
            "data": result["data"],
            "error": result["error"],
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {e}")
