import os
import googlemaps
from typing import Optional, Tuple

# Initialize the client with your Google Maps API key
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=API_KEY)

def get_lat_lon(location: str) -> Optional[Tuple[float, float]]:
    """
    Fetch the latitude and longitude of a named location using the Google Maps Geocoding API.
    Args:
        location (str): A named location (e.g., "Washington, DC").
    Returns:
        Optional[Tuple[float, float]]: A tuple of (latitude, longitude),
        or None if the location could not be found or an error occurs.
    """
    try:

        geocode_result = gmaps.geocode(location)

        if not geocode_result:
            print(f"No results found for location: '{location}'")
            return None

        lat_lon = geocode_result[0].get("geometry").get("location")
        return (lat_lon["lat"], lat_lon["lng"])

    except Exception as e:
        print(f"Request error fetching location data: {e}")
        return None

def check_location_in_us(location: str) -> bool:
    """
    Checks if a written location string is inside the United States.
    """
    try:
        # Perform forward geocoding
        geocode_result = gmaps.geocode(location)
        
        if not geocode_result:
            print(f"No results found for location: '{location}'")
            return False
            
        # Parse the first matching result
        address_components = geocode_result[0].get("address_components", [])
        return is_inside_us(address_components)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def is_inside_us(address_components):
    """
    Helper function to parse Google Maps address components for the United States.
    """
    for component in address_components:
        types = component.get("types", [])
        # Look specifically for the country component
        if "country" in types:
            short_name = component.get("short_name", "")
            long_name = component.get("long_name", "")
            return short_name == "US" or long_name == "United States"
    return False