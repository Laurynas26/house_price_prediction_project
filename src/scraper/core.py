import logging
from typing import Dict, Any
from src.scraper.scraper import FundaScraper
from src.scraper.utils import can_scrape, save_results
from src.scraper.logging_config import setup_logging
import os
import traceback


def scrape_listing(
    url: str,
    selectors_path: str = "config/selectors.json",
    headless: bool = True,
) -> Dict[str, Any]:
    """
    Scrape a single Funda listing URL and return parsed results.
    Handles errors gracefully for API usage.
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

        # Try to run scraper and catch any selenium/browser errors
        try:
            results = scraper.run()
        except Exception as e:
            logging.error(f"[ERROR] Selenium scrape failed for {url}: {e}")
            logging.error(traceback.format_exc())
            return {
                "success": False,
                "url": url,
                "data": None,
                "error": f"Selenium error: {str(e)}",
            }

        logging.info("[DEBUG] Scraper raw results:")
        for k, v in results.items():
            logging.info(f"    {k}: {v}")

        # Correct output dir based on environment
        output_dir = (
            "/tmp/api_scrapes"
            if os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
            else "data/api_scrapes"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save HTML & results if available
        if scraper.soup:
            try:
                save_results(
                    scraper.soup.prettify(),
                    results,
                    url,
                    output_dir=output_dir,
                )
            except Exception as e:
                logging.warning(f"[WARN] Failed saving results: {e}")

        return {"success": True, "url": url, "data": results, "error": None}

    except Exception as e:
        logging.error(f"[ERROR] scrape_listing failed for {url}: {e}")
        logging.error(traceback.format_exc())
        return {"success": False, "url": url, "data": None, "error": str(e)}
