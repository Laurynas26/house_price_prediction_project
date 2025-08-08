# Funda.nl Scraper Selectors

## Basic Info
- Price: `div.flex.flex-col.text-xl div.flex.gap-2.font-bold span`
- Address: `h1[data-global-id] > span.block.font-bold`
- Postal Code: `h1[data-global-id] > span.text-neutral-40`
- Neighborhood: `h1[data-global-id] > a`
- Status: `div.flex.items-baseline > div.bg-red-70`

## Summary Info (via dt/dd)
- Contribution VvE: label = "Bijdrage VvE"
- Year of Construction: label = "Bouwjaar"
- Roof Type: label = "Soort dak"
- Living Area: label = "Wonen"
- External Storage: label = "Externe bergruimte"
- Balcony: label = "Balkon/dakterras"
- Number of Rooms: label = "Aantal kamers"
- Bathrooms: label = "Aantal badkamers"
- Located On: label = "Gelegen op"
- Facilities: label = "Voorzieningen"

## Special Sections
- Neighborhood Details container: `div.flex.justify-between.border-b`
- Cadastral Info container: `div[data-testid="category-cadastral"]`
- Outdoor Features container: `div[data-testid="category-buitenruimte"]`
