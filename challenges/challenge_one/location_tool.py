import os
import requests
from typing import Optional, Tuple

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
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is not set.")

        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": location,
            "key": api_key
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if data["status"] != "OK":
            print(f"Geocoding API error: {data['status']}")
            return None

        lat_lon = data["results"][0]["geometry"]["location"]
        return (lat_lon["lat"], lat_lon["lng"])

    except requests.exceptions.RequestException as e:
        print(f"Request error fetching location data: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing location data: {e}")
        return None