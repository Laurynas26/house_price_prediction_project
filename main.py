import time
import logging
from src.scraper.scraper import FundaScraper
from src.scraper.utils import can_scrape, save_results
from src.scraper.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()

    urls_path = "config/house_pages_scraped.txt"
    selectors_path = "config/selectors.json"

    max_retries = 3

    with open(urls_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    # Only Amsterdam apartments first
    amsterdam_urls = [url for url in urls if "/koop/amsterdam/appartement" in url]

    amsterdam_urls = amsterdam_urls[:5651]

    for url in amsterdam_urls:
        if can_scrape(url):
            logging.info(f"Scraping allowed for {url}")

            for attempt in range(1, max_retries + 1):
                scraper = FundaScraper(url, selectors_path=selectors_path)
                results = scraper.run()

                # Check if all parsed values are "N/A"
                if all(value == "N/A" for value in results.values()):
                    logging.warning(
                        f"Attempt {attempt} for {url} returned all N/A. Retrying..."
                    )
                    time.sleep(3)  # wait before retrying
                else:
                    if scraper.soup:
                        save_results(scraper.soup.prettify(), results, url)
                    else:
                        logging.warning(f"No HTML soup extracted for {url}")
                    break  # exit retry loop on success
            else:
                logging.error(
                    f"Failed to get valid data from {url} after {max_retries} attempts."
                )

            time.sleep(2)
        else:
            logging.warning(f"Scraping not allowed for {url}")
