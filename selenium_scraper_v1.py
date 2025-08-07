from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

url_2 = "https://www.funda.nl/detail/koop/amsterdam/appartement-bilderdijkkade-52-b/43944051/"
url = "https://www.funda.nl/detail/koop/amsterdam/appartement-marcantilaan-282/43082096/"
options = Options()
options.headless = True

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

driver.get(url)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# Price
price_elem = soup.select_one('div.flex.flex-col.text-xl div.flex.gap-2.font-bold span')
price = price_elem.get_text(strip=True) if price_elem else "N/A"

# Address
address_elem = soup.select_one('h1[data-global-id] > span.block.font-bold')
postal_elem = soup.select_one('h1[data-global-id] > span.text-neutral-40')
neighborhood_elem = soup.select_one('h1[data-global-id] > a')

address = address_elem.get_text(strip=True) if address_elem else "N/A"
postal_code = postal_elem.get_text(strip=True) if postal_elem else "N/A"
neighborhood = neighborhood_elem.get_text(strip=True) if neighborhood_elem else "N/A"

# Status
status_elem = soup.select_one('div.flex.items-baseline > div.bg-red-70')
status = status_elem.get_text(strip=True) if status_elem else "N/A"

# Size, Bedrooms, Energy Label
all_lis = soup.find_all('li', class_='flex')
size = bedrooms = energy_label = "N/A"
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

# Additional details from dd elements (all with classes 'flex items-center ...')
dd_elements = soup.select('dd.flex.items-center.border-b')

# Map to hold label -> value for easy extraction
additional_info = {
    "Contribution": "N/A",
    "Apartment or House": "N/A",
    "Built Year": "N/A",
    "Roof Type": "N/A",
    "Outside Area": "N/A",
    "External Storage": "N/A",
    "Isolation": "N/A",
    "Heating (simple)": "N/A",
    "Ownership Situation": "N/A",
    "Location Description": "N/A",
    "Balcony": "N/A",
    "Shed Storage": "N/A",
    "Parking": "N/A",
}

# Helper: keywords to identify fields in the dd span text
keywords = {
    "Contribution": ["per maand", "bijdrage"],
    "Apartment or House": ["woning", "appartement", "huis", "bovenwoning"],
    "Built Year": ["19", "20"],  # Year will likely start with 19xx or 20xx
    "Roof Type": ["dak", "dakbedekking"],
    "Outside Area": ["m²", "buiten"],
    "External Storage": ["berging", "m²", "bijkeuken", "box"],
    "Isolation": ["isolatie", "glas", "geïsoleerd"],
    "Heating (simple)": ["cv-ketel", "gas gestookt", "verwarming"],
    "Ownership Situation": ["erfpacht", "eigendom", "eigendomssituatie"],
    "Location Description": ["rustige weg", "water", "woonwijk", "ligging", "uitzicht"],
    "Balcony": ["balkon"],
    "Shed Storage": ["box", "berging", "schuur"],
    "Parking": ["parkeren", "parkeervergunning"],
}

for dd in dd_elements:
    text = dd.get_text(strip=True)
    text_lower = text.lower()
    for key, kw_list in keywords.items():
        if any(kw in text_lower for kw in kw_list):
            # Assign the text (or you can refine to just part of it)
            additional_info[key] = text
            break

# Neighborhood details (inhabitants, families with children, avg price per m²)
neigh_details = {
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
            neigh_details["Inhabitants in neighborhood"] = value
        elif "gezin met kinderen" in label:
            neigh_details["Families with children"] = value
        elif "gem. vraagprijs / m²" in label:
            neigh_details["Price per m² in neighborhood"] = value

# Print all
print(f"Price: {price}")
print(f"Address: {address}")
print(f"Postal Code: {postal_code}")
print(f"Neighborhood: {neighborhood}")
print(f"Status: {status}")
print(f"Size: {size}")
print(f"Bedrooms: {bedrooms}")
print(f"Energy Label: {energy_label}")

for key, val in additional_info.items():
    print(f"{key}: {val}")

for key, val in neigh_details.items():
    print(f"{key}: {val}")


built_year = "N/A"
dt_elements = soup.select('dt.pt-2.pr-4.pb-1.font-normal.text-neutral-60')

for dt in dt_elements:
    if dt.get_text(strip=True) == "Bouwjaar":
        # The <dd> is the next sibling of this <dt>
        dd = dt.find_next_sibling('dd')
        if dd:
            span_year = dd.select_one('span.mr-2')
            if span_year:
                built_year = span_year.get_text(strip=True)
        break

print(f"Built Year: {built_year}")


driver.quit()
