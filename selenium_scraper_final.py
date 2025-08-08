from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re

def setup_driver(headless=True):
    options = Options()
    options.headless = headless
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def get_soup_from_url(driver, url):
    driver.get(url)
    html = driver.page_source
    return BeautifulSoup(html, 'html.parser')

def extract_text(selector, soup, default="N/A"):
    elem = soup.select_one(selector)
    return elem.get_text(strip=True) if elem else default

def extract_dd_by_dt_label(soup, label_text, exact_match=False, default="N/A"):
    """Extract text from <dd> next to <dt> matching label_text."""
    if exact_match:
        dt = soup.find('dt', string=lambda s: s and s.strip() == label_text)
    else:
        dt = soup.find('dt', string=lambda s: s and label_text in s)
    if dt:
        dd = dt.find_next_sibling('dd')
        if dd:
            return dd.get_text(strip=True)
    return default

def parse_size_bedrooms_energy(soup):
    size = bedrooms = energy_label = "N/A"
    all_lis = soup.find_all('li', class_='flex')
    for li in all_lis:
        label = li.find('span', class_='ml-1')
        value = li.find('span', class_='md:font-bold')
        if not label or not value:
            continue
        label_text = label.get_text(strip=True).lower()
        value_text = value.get_text(strip=True)
        if 'wonen' in label_text:
            size = value_text
        elif 'slaapkamers' in label_text:
            bedrooms = value_text
        elif 'energielabel' in label_text:
            energy_label = value_text
    return size, bedrooms, energy_label

def parse_neighborhood_details(soup):
    details = {
        "Inhabitants in neighborhood": "N/A",
        "Families with children": "N/A",
        "Price per m² in neighborhood": "N/A",
    }
    divs = soup.select('div.flex.justify-between.border-b')
    for div in divs:
        dt = div.find('dt')
        dd = div.find('dd')
        if dt and dd:
            label = dt.get_text(strip=True).lower()
            value = dd.get_text(strip=True)
            if "inwoners" in label:
                details["Inhabitants in neighborhood"] = value
            elif "gezin met kinderen" in label:
                details["Families with children"] = value
            elif "gem. vraagprijs / m²" in label:
                details["Price per m² in neighborhood"] = value
    return details

def parse_rooms_info(info_text):
    nr_rooms = "N/A"
    if info_text:
        match = re.search(r"(\d+)\s*kamers?", info_text.lower())
        if match:
            nr_rooms = match.group(1)
    return nr_rooms

def parse_bathrooms_and_toilets(info_text):
    bathrooms = toilets = "N/A"
    if info_text:
        bath_match = re.search(r"(\d+)\s*badkamer(s)?", info_text.lower())
        if bath_match:
            bathrooms = bath_match.group(1)
        toilet_match = re.search(r"(\d+)\s*apart(e)? toilet(ten)?", info_text.lower())
        if toilet_match:
            toilets = toilet_match.group(1)
    return bathrooms, toilets

def parse_cadastral_info(soup):
    cadastral_section = soup.find('div', attrs={'data-testid': 'category-cadastral'})
    ownership_situations = []
    charges = []
    cadastral_parcels = []
    added_parcels = set()

    if cadastral_section:
        dts = cadastral_section.find_all('dt')
        for dt in dts:
            dt_text = dt.get_text(strip=True)
            dd = dt.find_next_sibling('dd')
            if not dd:
                continue
            dd_text = dd.get_text(strip=True)

            if "Eigendomssituatie" in dt_text:
                ownership_situations.append(dd_text)
            elif "Lasten" in dt_text:
                charges.append(dd_text)
            elif "AMSTERDAM" in dt_text:
                if dt_text not in added_parcels:
                    added_parcels.add(dt_text)
                    cadastral_parcels.append({
                        'parcel': dt_text,
                        'link': dd.find('a')['href'] if dd.find('a') else None
                    })
    return cadastral_parcels, ownership_situations, charges

def parse_outdoor_features(soup):
    outdoor_section = soup.find('div', attrs={'data-testid': 'category-buitenruimte'})
    features = {
        'Ligging': None,
        'Tuin': None,
        'Achtertuin': None,
        'Ligging tuin': None
    }
    if outdoor_section:
        dts = outdoor_section.find_all('dt')
        for dt in dts:
            label = dt.get_text(strip=True)
            dd = dt.find_next_sibling('dd')
            if dd and label in features:
                features[label] = dd.get_text(strip=True)
    return features

def main(url):
    driver = setup_driver()
    soup = get_soup_from_url(driver, url)

    price = extract_text('div.flex.flex-col.text-xl div.flex.gap-2.font-bold span', soup)
    address = extract_text('h1[data-global-id] > span.block.font-bold', soup)
    postal_code = extract_text('h1[data-global-id] > span.text-neutral-40', soup)
    neighborhood = extract_text('h1[data-global-id] > a', soup)
    status = extract_text('div.flex.items-baseline > div.bg-red-70', soup)

    size, bedrooms, energy_label = parse_size_bedrooms_energy(soup)
    neigh_details = parse_neighborhood_details(soup)

    contribution = extract_dd_by_dt_label(soup, "Bijdrage VvE")
    year_of_construction = extract_dd_by_dt_label(soup, "Bouwjaar")
    roof_type = extract_dd_by_dt_label(soup, "Soort dak")
    living_area = extract_dd_by_dt_label(soup, "Wonen")
    external_storage = extract_dd_by_dt_label(soup, "Externe bergruimte")
    balcony = extract_dd_by_dt_label(soup, "Balkon/dakterras")
    rooms_info = extract_dd_by_dt_label(soup, "Aantal kamers")
    nr_rooms = parse_rooms_info(rooms_info)
    bathrooms_info = extract_dd_by_dt_label(soup, "Aantal badkamers")
    bathrooms, toilets = parse_bathrooms_and_toilets(bathrooms_info)
    located_on = extract_dd_by_dt_label(soup, "Gelegen op")

    facilities = extract_dd_by_dt_label(soup, "Voorzieningen")

    cadastral_parcels, ownership_situations, charges = parse_cadastral_info(soup)
    outdoor_features = parse_outdoor_features(soup)

    # Print all results
    print(f"Price: {price}")
    print(f"Address: {address}")
    print(f"Postal Code: {postal_code}")
    print(f"Neighborhood: {neighborhood}")
    print(f"Status: {status}")
    print(f"Size: {size}")
    print(f"Bedrooms: {bedrooms}")
    print(f"Energy Label: {energy_label}")
    print(f"Contribution VvE: {contribution}")
    print(f"Year of Construction: {year_of_construction}")
    print(f"Roof Type: {roof_type}")
    print(f"Living Area: {living_area}")
    print(f"External Storage: {external_storage}")
    print(f"Balcony: {balcony}")
    print(f"Number of Rooms: {nr_rooms}")
    print(f"Bathrooms: {bathrooms}")
    print(f"Separate Toilets: {toilets}")
    print(f"Located On: {located_on}")
    print(f"Facilities: {facilities}")
    print("Neighborhood Details:")
    for key, val in neigh_details.items():
        print(f"  {key}: {val}")
    print("Cadastral Parcels:", cadastral_parcels)
    print("Ownership Situations:", ownership_situations)
    print("Charges:", charges)
    print("Outdoor Features:", outdoor_features)

    driver.quit()


if __name__ == "__main__":
    test_url = "https://www.funda.nl/detail/koop/amsterdam/appartement-bilderdijkkade-52-b/43944051/"
    main(test_url)
