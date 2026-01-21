# `src.scraper`

Scraper module for extracting property listings from **Funda.nl**.
Handles scraping, data saving, logging, and robots.txt compliance.

---

## Overview

| File | Purpose |
|------|---------|
| `core.py` | Orchestrates scraping a URL, error handling, and saving results (local or S3). |
| `scraper.py` | `FundaScraper` class: Selenium + BeautifulSoup scraper using CSS selectors from `config/selectors.json`. |
| `logging_config.py` | Configures logging for local dev (file + console) or AWS Lambda (CloudWatch). |
| `utils.py` | Utilities: robots.txt check, unique ID generation, and saving HTML/JSON results. |
| `scrape_funda_url_for_data.py` | Standalone script to scrape multiple URLs listed in `config/house_pages_scraped.txt` with retries and logging. |

---

## Key Features

- Config-driven parsing with CSS selectors.
- Robust scraping with error handling.
- Save raw HTML and parsed JSON locally or to S3.
- AWS Lambda-ready logging.
- Robots.txt compliance checks.
- Supports batch scraping with retry logic via `scrape_funda_url_for_data.py`.

---

## Usage

### Single URL

One can run `scrape_funda_url_for_data.py` changing this line of code
`amsterdam_urls = amsterdam_urls[:5651]` to take 1 listing.

Otherwise, example code:

```python
from src.scraper.core import scrape_listing

url = "https://www.funda.nl/en/koop/amsterdam/appartement-12345678/"
result = scrape_listing(url, selectors_path="config/selectors.json", headless=True)

if result["success"]:
    print(result["data"])
else:
    print("Error:", result["error"])
