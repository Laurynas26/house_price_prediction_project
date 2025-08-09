import time
import logging
from src.scraper.scraper import FundaScraper
from src.scraper.utils import can_scrape, save_results
from src.scraper.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()

    urls_path = "config/house_pages.txt"
    selectors_path = "config/selectors.json"

    with open(urls_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        if can_scrape(url):
            logging.info(f"Scraping allowed for {url}")
            scraper = FundaScraper(url, selectors_path)
            results = scraper.run()

            if scraper.soup:
                save_results(scraper.soup.prettify(), results, url)
            else:
                logging.warning(f"No HTML soup extracted for {url}")

            time.sleep(2)
        else:
            logging.warning(f"Scraping not allowed for {url}")
