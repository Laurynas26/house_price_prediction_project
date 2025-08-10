from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time
import os

def fetch_xml_with_selenium(url: str):
    options = Options()
    options.headless = False
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1200,800")
    service = Service()  # or provide chromedriver path if needed
    
    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.get(url)
        time.sleep(5)  # wait for page to load
        page_source = driver.page_source
        
        soup = BeautifulSoup(page_source, 'html.parser')
        xml_div = soup.find('div', id='webkit-xml-viewer-source-xml')
        if not xml_div:
            raise Exception(f"XML container div not found on {url}")
        
        xml_str = ''.join(str(x) for x in xml_div.contents)
        root = ET.fromstring(xml_str)
        return root
    finally:
        driver.quit()

def get_sitemap_urls(sitemap_index_url):
    root = fetch_xml_with_selenium(sitemap_index_url)
    # Namespace handling
    ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    
    urls = []
    for sitemap in root.findall('ns:sitemap', ns):
        loc = sitemap.find('ns:loc', ns)
        if loc is not None:
            urls.append(loc.text)
    return urls

def get_urls_from_sitemap(sitemap_url):
    root = fetch_xml_with_selenium(sitemap_url)
    ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    
    urls = []
    for url in root.findall('ns:url', ns):
        loc = url.find('ns:loc', ns)
        if loc is not None:
            urls.append(loc.text)
    return urls

if __name__ == "__main__":
    sitemap_index_url = "https://www.funda.nl/sitemap_index.xml"
    
    # Step 1: Get all inner sitemap URLs
    inner_sitemaps = get_sitemap_urls(sitemap_index_url)
    print(f"Found {len(inner_sitemaps)} inner sitemaps")
    
    all_urls = []
    # Step 2 & 3: For each inner sitemap, get actual URLs
    for sitemap_url in inner_sitemaps:
        print(f"Fetching URLs from sitemap: {sitemap_url}")
        urls = get_urls_from_sitemap(sitemap_url)
        print(f" - Found {len(urls)} URLs")
        all_urls.extend(urls)
    
    print(f"Total URLs collected: {len(all_urls)}")

    # Make sure folder 'config' exists
    os.makedirs("config", exist_ok=True)

    # Save URLs to the file
    with open("config/house_pages_scraped.txt", "w", encoding="utf-8") as f:
        for url in all_urls:
            f.write(url + "\n")

    print("URLs saved to config/house_pages_scraped.txt")
