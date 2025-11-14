import logging
from typing import Dict, Any
from src.scraper.scraper import FundaScraper
from src.scraper.utils import can_scrape, save_results
from src.scraper.logging_config import setup_logging
import os

def scrape_listing(
    url: str,
    selectors_path: str = "config/selectors.json",
    headless: bool = True,
) -> Dict[str, Any]:
    """
    Scrape a single Funda listing URL and return parsed results.
    Handles errors gracefully for API usage.

    Args:
        url (str): Funda listing URL.
        selectors_path (str): Path to JSON selectors config.
        headless (bool): Run Chrome in headless mode.

    Returns:
        dict: {
            "success": bool,
            "url": str,
            "data": dict (parsed results) or None,
            "error": str or None
        }
    """
    setup_logging()
    logging.info(f"[DEBUG] Starting scrape for {url}")

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
        results = scraper.run()

        # Log each field for debugging
        logging.info("[DEBUG] Scraper raw results:")
        for k, v in results.items():
            logging.info(f"    {k}: {v}")

        # ----------- FIX: choose correct output dir -----------
        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            output_dir = "/tmp/api_scrapes"
        else:
            output_dir = "data/api_scrapes"
        os.makedirs(output_dir, exist_ok=True)
        # --------------------------------------------------------

        # Optionally save HTML & results for traceability
        if scraper.soup:
            save_results(
                scraper.soup.prettify(),
                results,
                url,
                output_dir=output_dir,
            )

        return {"success": True, "url": url, "data": results, "error": None}

    except Exception as e:
        logging.error(f"[ERROR] Scraping failed for {url}: {e}")
        return {"success": False, "url": url, "data": None, "error": str(e)}
