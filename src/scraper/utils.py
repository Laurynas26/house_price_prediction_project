import os
import json
import logging
import hashlib
from datetime import datetime
from urllib.robotparser import RobotFileParser


def can_scrape(url: str, user_agent: str = "*") -> bool:
    """
    Check robots.txt to verify if scraping the URL is allowed.

    Args:
        url (str): The URL to check.
        user_agent (str, optional): User agent to check permissions for. Defaults to "*".

    Returns:
        bool: True if scraping is allowed, False otherwise.
    """
    rp = RobotFileParser()
    base_url_parts = url.split("/", 3)[
        :3
    ]  # e.g. ['https:', '', 'www.funda.nl']
    base_url = "/".join(base_url_parts)
    rp.set_url(base_url + "/robots.txt")
    rp.read()
    return rp.can_fetch(user_agent, url)


def generate_id_from_url(url: str) -> str:
    """
    Generate a unique ID based on the URL and current timestamp.

    Args:
        url (str): The URL to generate ID from.

    Returns:
        str: Unique ID string.
    """
    hash_id = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{hash_id}_{timestamp}"


def save_results(
    raw_html: str, parsed_data: dict, url: str, output_dir: str = "data"
):
    """
    Save raw HTML and parsed JSON into separate folders inside output_dir.

    Args:
        raw_html (str): The raw HTML content as a string.
        parsed_data (dict): The parsed results.
        url (str): The source URL (used for unique ID generation).
        output_dir (str, optional): Base directory to store results. Defaults to "data".
    """
    html_dir = os.path.join(output_dir, "raw_html")
    json_dir = os.path.join(output_dir, "parsed_json")
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    file_id = generate_id_from_url(url)

    if raw_html:
        html_path = os.path.join(html_dir, f"{file_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(raw_html)
        logging.info(f"Saved HTML to {html_path}")
    else:
        logging.warning(f"No HTML to save for {url}")

    if parsed_data:
        json_path = os.path.join(json_dir, f"{file_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved JSON to {json_path}")
    else:
        logging.warning(f"No parsed data to save for {url}")
