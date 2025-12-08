import logging
from typing import Dict, Any
from src.scraper.scraper import FundaScraper
from src.scraper.utils import can_scrape, save_results
from src.scraper.logging_config import setup_logging
import os
import traceback
import uuid
import json
import boto3

# Setup logging once per scrape
setup_logging()

# S3 client
s3 = boto3.client("s3")
BUCKET_NAME = os.getenv("S3_BUCKET")


def generate_job_id() -> str:
    """Generate a unique ID for each scrape."""
    return str(uuid.uuid4())


def scrape_listing(
    url: str,
    selectors_path: str = "config/selectors.json",
    headless: bool = True,
) -> Dict[str, Any]:
    """
    Scrape a single Funda listing URL and return parsed results.
    Handles errors gracefully for API usage.
    Saves HTML and parsed JSON either locally or to S3 if BUCKET_NAME is set.
    """
    logging.info(f"[DEBUG] Starting scrape for {url}")

    # Check robots.txt
    if not can_scrape(url):
        warning_msg = f"Scraping disallowed by robots.txt for {url}"
        logging.warning(f"[DEBUG] {warning_msg}")
        return {
            "success": False,
            "url": url,
            "data": None,
            "error": warning_msg,
        }

    try:
        scraper = FundaScraper(
            url, selectors_path=selectors_path, headless=headless
        )

        # Run the scraper
        results = scraper.run()

        logging.info("[DEBUG] Scraper raw results:")
        for k, v in results.items():
            logging.info(f"    {k}: {v}")

        # Save results
        if scraper.soup:
            job_id = generate_job_id()
            if BUCKET_NAME:
                # Save HTML & JSON to S3
                html_key = f"scraped_raw_html/{job_id}.html"
                json_key = f"scraped_parsed_json/{job_id}.json"
                try:
                    s3.put_object(
                        Bucket=BUCKET_NAME,
                        Key=html_key,
                        Body=scraper.soup.prettify(),
                        ContentType="text/html",
                    )
                    s3.put_object(
                        Bucket=BUCKET_NAME,
                        Key=json_key,
                        Body=json.dumps(results),
                        ContentType="application/json",
                    )
                    logging.info(
                        "[INFO] Saved HTML and JSON to S3:"
                        f"{html_key}, {json_key}"
                    )
                except Exception as e:
                    logging.warning(f"[WARN] Failed to save to S3: {e}")
            else:
                # Save locally
                output_dir = "data/api_scrapes"
                os.makedirs(output_dir, exist_ok=True)
                try:
                    save_results(
                        scraper.soup.prettify(),
                        results,
                        url,
                        output_dir=output_dir,
                    )
                    logging.info(
                        f"[INFO] Saved HTML and JSON locally at {output_dir}"
                    )
                except Exception as e:
                    logging.warning(f"[WARN] Failed saving locally: {e}")

        return {"success": True, "url": url, "data": results, "error": None}

    except Exception as e:
        logging.error(f"[ERROR] scrape_listing failed for {url}: {e}")
        logging.error(traceback.format_exc())
        return {"success": False, "url": url, "data": None, "error": str(e)}
