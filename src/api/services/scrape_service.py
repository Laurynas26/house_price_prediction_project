import uuid
import os
from src.scraper.core import scrape_listing

# Get S3 bucket from environment if running on EC2 or elsewhere
BUCKET_NAME = os.getenv("S3_BUCKET")


def generate_job_id() -> str:
    return str(uuid.uuid4())


def scrape_and_store(url: str, headless: bool = True):
    """
    Scrape a Funda listing and store results.
    Returns:
        - job_id
        - paths/keys (dict with 'html' and 'json')
        - error (if any)
    """
    result = scrape_listing(url, headless=headless)

    if not result["success"]:
        return None, None, result["error"]

    job_id = generate_job_id()

    if BUCKET_NAME:
        # Assume S3 storage inside scrape_listing
        html_key = f"scraped_raw_html/{job_id}.html"
        json_key = f"scraped_parsed_json/{job_id}.json"
        return job_id, {"html": html_key, "json": json_key}, None
    else:
        # Local storage paths
        html_path = os.path.join(
            "data/api_scrapes/scraped_raw_html", f"{job_id}.html"
        )
        json_path = os.path.join(
            "data/api_scrapes/scraped_parsed_json", f"{job_id}.json"
        )
        return job_id, {"html": html_path, "json": json_path}, None
