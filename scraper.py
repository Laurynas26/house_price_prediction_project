import requests
from bs4 import BeautifulSoup

# Example Funda detail page URL (replace with a real one from funda.nl)
url = "https://www.funda.nl/koop/amsterdam/huis-1234567-example-street-1/"

# Set headers with a user-agent to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

def scrape_funda_detail(url):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch page, status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract price (example selector, update after inspection)
    price_elem = soup.select_one('span.object-header-price')
    price = price_elem.get_text(strip=True) if price_elem else "N/A"

    # Extract size (m²) - example selector, adjust as needed
    size_elem = soup.select_one('li.object-kenmerken__item--surface-area span.object-kenmerken__value')
    size = size_elem.get_text(strip=True) if size_elem else "N/A"

    # Extract number of rooms - example selector, adjust as needed
    rooms_elem = soup.select_one('li.object-kenmerken__item--number-of-rooms span.object-kenmerken__value')
    rooms = rooms_elem.get_text(strip=True) if rooms_elem else "N/A"

    # Print extracted data
    print(f"Price: {price}")
    print(f"Size: {size}")
    print(f"Rooms: {rooms}")

    return {
        "price": price,
        "size": size,
        "rooms": rooms
    }

if __name__ == "__main__":
    scrape_funda_detail(url)
