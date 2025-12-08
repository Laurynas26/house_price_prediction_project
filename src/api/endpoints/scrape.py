from fastapi import APIRouter, HTTPException, Query
from src.api.services.scrape_service import scrape_and_store

router = APIRouter()

@router.post("/")
def scrape_and_return(
    url: str = Query(..., description="Funda listing URL"), 
    headless: bool = True
):
    """
    Scrape a Funda listing and store parsed structured data
    with debug info in S3.
    """
    job_id, s3_key, error = scrape_and_store(url, headless)

    if error:
        raise HTTPException(status_code=400, detail=error)

    return {
        "job_id": job_id,
        "s3_key": s3_key,
        "status": "stored"
    }