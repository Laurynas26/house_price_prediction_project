import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
from typing import Tuple, Dict, List, Optional, Any
import json
import os


class FundaScraper:
    """
    Scraper for Funda.nl real estate listings.

    Uses Selenium to load the listing page, waits for key elements, then parses
    content using BeautifulSoup. Parsing logic is driven by a configurable set
    of CSS selectors defined in a JSON file.

    Attributes
    ----------
    url : str
        Target Funda listing URL.
    headless : bool
        Whether Chrome runs in headless mode.
    driver : webdriver.Chrome or None
        Selenium WebDriver instance (initialized in `setup_driver`).
    soup : BeautifulSoup or None
        Parsed HTML tree of the loaded page.
    results : dict
        Dictionary storing all parsed results.
    selectors : dict
        Configuration mapping of element selectors loaded from JSON.
    """

    def __init__(
        self,
        url: str,
        headless: bool = True,
        selectors_path: str = "config/selectors.json",
    ) -> None:
        """
        Initialize the scraper with the target URL and browser mode.

        Args:
            url (str): The URL of the listing to scrape.
            headless (bool, optional): Whether to run Chrome in headless mode.
            Defaults to True.
        """
        self.url: str = url
        self.headless: bool = headless
        self.driver: Optional[webdriver.Chrome] = None
        self.soup: Optional[BeautifulSoup] = None
        self.results: Dict[str, Any] = {}

        # Auto-locate selectors.json relative to this file
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        selectors_path = os.path.join(project_root, "config", "selectors.json")

        if not os.path.exists(selectors_path):
            raise FileNotFoundError(
                f"Selectors file not found at: {selectors_path}"
            )

        with open(selectors_path, "r", encoding="utf-8") as f:
            self.selectors = json.load(f)

    def setup_driver(self) -> None:
        """
        Set up Chrome WebDriver with proper options depending on environment.
        Works both locally and on AWS Lambda.
        """
        options = Options()
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-setuid-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--log-level=3")

        if os.environ.get("LAMBDA_TASK_ROOT"):
            # Lambda-specific options
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")
        else:
            # Local
            if self.headless:
                options.add_argument("--headless=new")

        try:
            service = Service(ChromeDriverManager().install())
            service.log_output = os.devnull  # silence Chrome logs
            self.driver = webdriver.Chrome(service=service, options=options)
        except Exception as e:
            logging.error(f"Failed to start ChromeDriver: {e}")
            raise RuntimeError("ChromeDriver initialization failed.") from e

    def get_soup_from_url(self) -> None:
        """
        Load the page and parse its HTML with BeautifulSoup.
        """
        assert self.driver, "Driver is not initialized."
        try:
            self.driver.get(self.url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.selectors["basic"]["price"])
                )
            )
        except Exception as e:
            logging.warning(f"Timeout or error waiting for page elements: {e}")
        finally:
            html = self.driver.page_source
            self.soup = BeautifulSoup(html, "html.parser")

    def extract_text(self, selector: str, default: str = "N/A") -> str:
        """
        Extract text from the first element matching the CSS selector.

        Args:
            selector (str): CSS selector to find element.
            default (str, optional): Default text if element not found.
            Defaults to "N/A".

        Returns:
            str: Extracted text or default.
        """
        assert self.soup is not None, "Soup is not initialized."
        elem = self.soup.select_one(selector)
        return elem.get_text(strip=True) if elem else default

    def extract_dd_by_dt_label(
        self, label_text: str, exact_match: bool = False, default: str = "N/A"
    ) -> str:
        """
        Extract text from the <dd> element that follows
        a <dt> matching label_text.

        Args:
            label_text (str): Text label to find in <dt>.
            exact_match (bool, optional): Whether match must be exact.
            Defaults to False.
            default (str, optional): Default text if not found.
            Defaults to "N/A".

        Returns:
            str: Extracted text or default.
        """
        assert self.soup is not None, "Soup is not initialized."
        if exact_match:
            dt = self.soup.find(
                "dt", string=lambda s: s and s.strip() == label_text
            )
        else:
            dt = self.soup.find("dt", string=lambda s: s and label_text in s)
        if dt:
            dd = dt.find_next_sibling("dd")
            if dd:
                return dd.get_text(strip=True)
        return default

    def parse_size_bedrooms_energy(self) -> Tuple[str, str, str]:
        """
        Parse size, number of bedrooms, and energy label from the page.

        Returns:
            Tuple[str, str, str]: Size, bedrooms, and energy label.
        """
        assert self.soup is not None, "Soup is not initialized."
        size = bedrooms = energy_label = "N/A"

        config = self.selectors["size_bedrooms_energy"]
        list_item_class = config["list_item_class"]
        label_span_class = config["label_span_class"]
        value_span_class = config["value_span_class"]
        labels = config["labels"]

        all_lis = self.soup.find_all(
            "li", class_=list_item_class.split(".")[-1]
        )  # assume last class from list_item_class
        for li in all_lis:
            label = li.find("span", class_=label_span_class.split(".")[-1])
            value = li.find("span", class_=value_span_class.split(".")[-1])
            if not label or not value:
                continue
            label_text = label.get_text(strip=True).lower()
            value_text = value.get_text(strip=True)
            if labels["size"] in label_text:
                size = value_text
            elif labels["bedrooms"] in label_text:
                bedrooms = value_text
            elif labels["energy_label"] in label_text:
                energy_label = value_text
        return size, bedrooms, energy_label

    def parse_neighborhood_details(self) -> Dict[str, str]:
        """
        Parse neighborhood details such as inhabitants, families with children,
        and price per m².

        Returns:
            Dict[str, str]: Neighborhood details.
        """
        config = self.selectors["neighborhood_details"]
        container_selector = config["container_selector"]
        labels = config["labels"]

        details = {
            "Inhabitants in neighborhood": "N/A",
            "Families with children": "N/A",
            "Price per m² in neighborhood": "N/A",
        }
        divs = self.soup.select(container_selector)
        for div in divs:
            dt = div.find("dt")
            dd = div.find("dd")
            if dt and dd:
                label = dt.get_text(strip=True).lower()
                value = dd.get_text(strip=True)
                if labels["inhabitants"] in label:
                    details["Inhabitants in neighborhood"] = value
                elif labels["families_with_children"] in label:
                    details["Families with children"] = value
                elif labels["price_per_m2"] in label:
                    details["Price per m² in neighborhood"] = value
        return details

    def parse_rooms_info(self, info_text: Optional[str]) -> str:
        """
        Extract number of rooms from text.

        Args:
            info_text (Optional[str]): Text containing room info.

        Returns:
            str: Number of rooms or "N/A".
        """
        nr_rooms = "N/A"
        if info_text:
            match = re.search(r"(\d+)\s*kamers?", info_text.lower())
            if match:
                nr_rooms = match.group(1)
        return nr_rooms

    def parse_bathrooms_and_toilets(
        self, info_text: Optional[str]
    ) -> Tuple[str, str]:
        """
        Extract number of bathrooms and separate toilets from text.

        Args:
            info_text (Optional[str]): Text containing bathroom/toilet info.

        Returns:
            Tuple[str, str]: Number of bathrooms and toilets or "N/A".
        """
        bathrooms = toilets = "N/A"
        if info_text:
            bath_match = re.search(r"(\d+)\s*badkamer(s)?", info_text.lower())
            if bath_match:
                bathrooms = bath_match.group(1)
            toilet_match = re.search(
                r"(\d+)\s*apart(e)? toilet(ten)?", info_text.lower()
            )
            if toilet_match:
                toilets = toilet_match.group(1)
        return bathrooms, toilets

    def parse_cadastral_info(
        self,
    ) -> Tuple[List[Dict[str, Optional[str]]], List[str], List[str]]:
        """
        Parse cadastral parcels, ownership situations, and charges.

        Returns:
            Tuple[List[Dict[str, Optional[str]]], List[str], List[str]]:
                - cadastral parcels list with parcel and link
                - ownership situations list
                - charges list
        """
        assert self.soup is not None, "Soup is not initialized."
        config = self.selectors["cadastral_info"]

        cadastral_section = self.soup.find(
            "div", attrs={"data-testid": config["section_data_testid"]}
        )
        ownership_situations: List[str] = []
        charges: List[str] = []
        cadastral_parcels: List[Dict[str, Optional[str]]] = []
        added_parcels = set()

        if cadastral_section:
            dts = cadastral_section.find_all("dt")
            for dt in dts:
                dt_text = dt.get_text(strip=True)
                dd = dt.find_next_sibling("dd")
                if not dd:
                    continue
                dd_text = dd.get_text(strip=True)

                if config["ownership_situation_label"] in dt_text:
                    ownership_situations.append(dd_text)
                elif config["charges_label"] in dt_text:
                    charges.append(dd_text)
                elif config["parcel_key"] in dt_text:
                    if dt_text not in added_parcels:
                        added_parcels.add(dt_text)
                        cadastral_parcels.append(
                            {
                                "parcel": dt_text,
                                "link": (
                                    dd.find("a")["href"]
                                    if dd.find("a")
                                    else None
                                ),
                            }
                        )
        return cadastral_parcels, ownership_situations, charges

    def parse_outdoor_features(self) -> Dict[str, Optional[str]]:
        """
        Parse outdoor features like garden and location.

        Returns:
            Dict[str, Optional[str]]: Outdoor features with keys 'Ligging',
            'Tuin', 'Achtertuin', 'Ligging tuin'.
        """
        assert self.soup is not None, "Soup is not initialized."
        config = self.selectors["outdoor_features"]

        outdoor_section = self.soup.find(
            "div", attrs={"data-testid": config["section_data_testid"]}
        )
        features = {field: None for field in config["fields"]}
        if outdoor_section:
            dts = outdoor_section.find_all("dt")
            for dt in dts:
                label = dt.get_text(strip=True)
                dd = dt.find_next_sibling("dd")
                if dd and label in features:
                    features[label] = dd.get_text(strip=True)
        return features

    def run(self) -> Dict[str, Any]:
        """
        Run the scraper:
        setup driver, fetch page, parse all info, and quit driver.

        Returns:
            Dict[str, Any]: Dictionary with all scraped results.
        """
        try:
            logging.info(f"Starting scraper for URL: {self.url}")
            self.setup_driver()
            self.get_soup_from_url()

            # Wrap each parsing section if needed for robustness
            try:
                basic = self.selectors["basic"]
                self.results["price"] = self.extract_text(basic["price"])
                self.results["address"] = self.extract_text(basic["address"])
                self.results["postal_code"] = self.extract_text(
                    basic["postal_code"]
                )
                self.results["neighborhood"] = self.extract_text(
                    basic["neighborhood"]
                )
                self.results["status"] = self.extract_text(basic["status"])
            except Exception as e:
                logging.warning(f"Error parsing basic info: {e}")

            try:
                size, bedrooms, energy_label = (
                    self.parse_size_bedrooms_energy()
                )
                self.results["size"] = size
                self.results["bedrooms"] = bedrooms
                self.results["energy_label"] = energy_label
            except Exception as e:
                logging.warning(f"Error parsing size/bedrooms/energy: {e}")

            try:
                self.results["neighborhood_details"] = (
                    self.parse_neighborhood_details()
                )
            except Exception as e:
                logging.warning(f"Error parsing neighborhood details: {e}")

            # Continue with other parsing sections...
            dl_labels = self.selectors["dl_labels"]
            for label_key, parser in [
                ("contribution_vve", self.extract_dd_by_dt_label),
                ("year_of_construction", self.extract_dd_by_dt_label),
                ("roof_type", self.extract_dd_by_dt_label),
                ("living_area", self.extract_dd_by_dt_label),
                ("external_storage", self.extract_dd_by_dt_label),
                ("balcony", self.extract_dd_by_dt_label),
                (
                    "rooms_info",
                    lambda x: self.parse_rooms_info(
                        self.extract_dd_by_dt_label(x)
                    ),
                ),
                (
                    "bathrooms_info",
                    lambda x: self.parse_bathrooms_and_toilets(
                        self.extract_dd_by_dt_label(x)
                    ),
                ),
                ("located_on", self.extract_dd_by_dt_label),
                ("facilities", self.extract_dd_by_dt_label),
            ]:
                try:
                    value = parser(dl_labels[label_key])
                    if label_key == "bathrooms_info":
                        self.results["bathrooms"], self.results["toilets"] = (
                            value
                        )
                    elif label_key == "rooms_info":
                        self.results["nr_rooms"] = value
                    else:
                        self.results[label_key.replace("_info", "")] = value
                except Exception as e:
                    logging.warning(f"Error parsing {label_key}: {e}")

            try:
                cadastral_parcels, ownership_situations, charges = (
                    self.parse_cadastral_info()
                )
                self.results["cadastral_parcels"] = cadastral_parcels
                self.results["ownership_situations"] = ownership_situations
                self.results["charges"] = charges
            except Exception as e:
                logging.warning(f"Error parsing cadastral info: {e}")

            try:
                self.results["outdoor_features"] = (
                    self.parse_outdoor_features()
                )
            except Exception as e:
                logging.warning(f"Error parsing outdoor features: {e}")

            logging.info(f"Scraping successful for URL: {self.url}")
            self.results["success"] = True
            return self.results

        except Exception as e:
            logging.error(
                f"Error during scraping {self.url}: {e}", exc_info=True
            )
            return {
                "success": False,
                "url": self.url,
                "data": None,
                "error": str(e),
            }

        finally:
            if self.driver:
                self.driver.quit()
