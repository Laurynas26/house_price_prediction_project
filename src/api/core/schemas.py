from typing import Dict, Tuple, Any


# Expected input schema for a listing
# Format:
# field_name: (default_value, allowed_types)
EXPECTED_SCHEMA: Dict[str, Tuple[Any, tuple]] = {
    "price": (None, (int, float, str, type(None))),
    "contribution_vve": (None, (int, float, str, type(None))),
    "size": (None, (int, float, str, type(None))),
    "external_storage": (None, (int, float, str, type(None))),
    "year_of_construction": (None, (int, float, str, type(None))),
    "nr_rooms": (None, (int, float, str, type(None))),
    "bathrooms": (None, (int, float, str, type(None))),
    "toilets": (None, (int, float, str, type(None))),
    "bedrooms": (None, (int, float, str, type(None))),
    "facilities": ("", (str, list, type(None))),
    "outdoor_features": ({}, (dict, type(None))),
    "cadastral_parcels": ([], (list, type(None))),
    "ownership_situations": ([], (list, type(None))),
    "charges": ([], (list, type(None))),
    "postal_code": (None, (str, type(None))),
    "neighborhood_details": ({}, (dict, type(None))),
    "address": (None, (str, type(None))),
    "roof_type": (None, (str, type(None))),
    "status": (None, (str, type(None))),
    "ownership_type": (None, (str, type(None))),
    "location": (None, (str, type(None))),
    "energy_label": (None, (str, type(None))),
    "located_on": (None, (str, type(None))),
    "backyard": (None, (str, type(None))),
    "balcony": (None, (str, type(None))),
}


def build_listing_from_manual_input(manual_input: dict) -> dict:
    """
    Build a schema-safe listing dict from manual user input.

    Any missing fields are filled with schema defaults.
    Extra fields are ignored.
    """
    listing = {}

    for field, (default, _) in EXPECTED_SCHEMA.items():
        listing[field] = manual_input.get(field, default)

    return listing
