import logging
import time
from urllib.robotparser import RobotFileParser
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
from typing import Tuple, Dict, List, Optional, Any
import json 
import os


class FundaScraper:
    def __init__(self, url: str, headless: bool = True, selectors_path: str = "config/selectors.json") -> None:
        """
        Initialize the scraper with the target URL and browser mode.

        Args:
            url (str): The URL of the listing to scrape.
            headless (bool, optional): Whether to run Chrome in headless mode. Defaults to True.
        """
        self.url: str = url
        self.headless: bool = headless
        self.driver: Optional[webdriver.Chrome] = None
        self.soup: Optional[BeautifulSoup] = None
        self.results: Dict[str, Any] = {}

        # Load selectors JSON config
        if not os.path.exists(selectors_path):
            raise FileNotFoundError(f"Selectors file not found: {selectors_path}")
        with open(selectors_path, "r", encoding="utf-8") as f:
            self.selectors = json.load(f)

    def setup_driver(self) -> None:
        """
        Set up the Chrome WebDriver with options.
        """
        options = Options()
        options.headless = self.headless
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

    def get_soup_from_url(self) -> None:
        """
        Load the page and parse its HTML with BeautifulSoup.
        """
        assert self.driver is not None, "Driver is not initialized."
        self.driver.get(self.url)
        html = self.driver.page_source
        self.soup = BeautifulSoup(html, "html.parser")

    def extract_text(self, selector: str, default: str = "N/A") -> str:
        """
        Extract text from the first element matching the CSS selector.

        Args:
            selector (str): CSS selector to find element.
            default (str, optional): Default text if element not found. Defaults to "N/A".

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
        Extract text from the <dd> element that follows a <dt> matching label_text.

        Args:
            label_text (str): Text label to find in <dt>.
            exact_match (bool, optional): Whether match must be exact. Defaults to False.
            default (str, optional): Default text if not found. Defaults to "N/A".

        Returns:
            str: Extracted text or default.
        """
        assert self.soup is not None, "Soup is not initialized."
        if exact_match:
            dt = self.soup.find("dt", string=lambda s: s and s.strip() == label_text)
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


        all_lis = self.soup.find_all("li", class_=list_item_class.split('.')[-1])  # assume last class from list_item_class
        for li in all_lis:
            label = li.find("span", class_=label_span_class.split('.')[-1])
            value = li.find("span", class_=value_span_class.split('.')[-1])
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
        Parse neighborhood details such as inhabitants, families with children, and price per m².

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

    def parse_bathrooms_and_toilets(self, info_text: Optional[str]) -> Tuple[str, str]:
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

        cadastral_section = self.soup.find("div", attrs={"data-testid": config["section_data_testid"]})
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
                        cadastral_parcels.append({
                            "parcel": dt_text,
                            "link": dd.find("a")["href"] if dd.find("a") else None,
                        })
        return cadastral_parcels, ownership_situations, charges

    def parse_outdoor_features(self) -> Dict[str, Optional[str]]:
        """
        Parse outdoor features like garden and location.

        Returns:
            Dict[str, Optional[str]]: Outdoor features with keys 'Ligging', 'Tuin', 'Achtertuin', 'Ligging tuin'.
        """
        assert self.soup is not None, "Soup is not initialized."
        config = self.selectors["outdoor_features"]

        outdoor_section = self.soup.find("div", attrs={"data-testid": config["section_data_testid"]})
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
        Run the scraper: setup driver, fetch page, parse all info, and quit driver.

        Returns:
            Dict[str, Any]: Dictionary with all scraped results.
        """
        try:
            logging.info(f"Starting scraper for URL: {self.url}")
            self.setup_driver()
            self.get_soup_from_url()

            basic = self.selectors["basic"]
            self.results["price"] = self.extract_text(basic["price"])
            self.results["address"] = self.extract_text(basic["address"])
            self.results["postal_code"] = self.extract_text(basic["postal_code"])
            self.results["neighborhood"] = self.extract_text(basic["neighborhood"])
            self.results["status"] = self.extract_text(basic["status"])

            size, bedrooms, energy_label = self.parse_size_bedrooms_energy()
            self.results["size"] = size
            self.results["bedrooms"] = bedrooms
            self.results["energy_label"] = energy_label

            self.results["neighborhood_details"] = self.parse_neighborhood_details()

            dl_labels = self.selectors["dl_labels"]
            self.results["contribution"] = self.extract_dd_by_dt_label(dl_labels["contribution_vve"])
            self.results["year_of_construction"] = self.extract_dd_by_dt_label(dl_labels["year_of_construction"])
            self.results["roof_type"] = self.extract_dd_by_dt_label(dl_labels["roof_type"])
            self.results["living_area"] = self.extract_dd_by_dt_label(dl_labels["living_area"])
            self.results["external_storage"] = self.extract_dd_by_dt_label(dl_labels["external_storage"])
            self.results["balcony"] = self.extract_dd_by_dt_label(dl_labels["balcony"])

            rooms_info = self.extract_dd_by_dt_label(dl_labels["rooms_info"])
            self.results["nr_rooms"] = self.parse_rooms_info(rooms_info)

            bathrooms_info = self.extract_dd_by_dt_label(dl_labels["bathrooms_info"])
            bathrooms, toilets = self.parse_bathrooms_and_toilets(bathrooms_info)
            self.results["bathrooms"] = bathrooms
            self.results["toilets"] = toilets

            self.results["located_on"] = self.extract_dd_by_dt_label(dl_labels["located_on"])
            self.results["facilities"] = self.extract_dd_by_dt_label(dl_labels["facilities"])

            cadastral_parcels, ownership_situations, charges = self.parse_cadastral_info()
            self.results["cadastral_parcels"] = cadastral_parcels
            self.results["ownership_situations"] = ownership_situations
            self.results["charges"] = charges

            self.results["outdoor_features"] = self.parse_outdoor_features()
            logging.info(f"Scraping successful for URL: {self.url}")

        except Exception as e:
            print(f"Error during scraping: {e}")
        finally:
            if self.driver:
                self.driver.quit()
        return self.results


def can_scrape(url: str, user_agent: str = "*") -> bool:
    rp = RobotFileParser()
    base_url = url.split("/", 3)[:3]  # e.g. https://www.funda.nl
    base_url = "/".join(base_url)
    rp.set_url(base_url + "/robots.txt")
    rp.read()
    return rp.can_fetch(user_agent, url)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        )
    logging.info("Starting scraper script...")
    # urls = [
    #     "https://www.funda.nl/detail/koop/amsterdam/appartement-bilderdijkkade-52-b/43944051/",
    # ]
    with open("config/house_pages.txt", "r", encoding="utf-8") as f: 
        urls = [line.strip() for line in f if line.strip()]
    for url in urls:
        if can_scrape(url):
            logging.info(f"Scraping allowed for {url}")
            scraper = FundaScraper(url)
            results = scraper.run()
            for k, v in results.items():
                logging.info(f"{k}: {v}")
            time.sleep(2)
        else:
            logging.warning(f"Scraping not allowed for {url}")
