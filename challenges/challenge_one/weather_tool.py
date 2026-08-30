import requests
from typing import Optional, List, Dict

def get_extended_weather_forecast(lat: float, lon: float) -> Optional[List[Dict[str, str]]]:
    """
    Fetch the extended weather forecast from the U.S. National Weather Service API
    based on a given latitude and longitude.
    Args:
        lat (float): Latitude of the location (e.g., 38.8977).
        lon (float): Longitude of the location (e.g., -77.0365).
    Returns:
        Optional[List[Dict[str, str]]]: A list of forecast dictionaries
        Returns None if data is unavailable or an error occurs.
    """
    try:
        # Step 1: Get the forecast grid endpoint for the given coordinates
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {"User-Agent": "WeatherBot/1.0"}

        points_response = requests.get(points_url, headers=headers)
        points_response.raise_for_status()

        forecast_url = points_response.json()["properties"]["forecast"]

        # Step 2: Fetch the extended forecast from the grid endpoint
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()

        periods = forecast_response.json()["properties"]["periods"]

        # Step 3: Parse and return the relevant forecast fields
        forecast = [
            {
                "name": period["name"],
                "temperature": f"{period['temperature']}°{period['temperatureUnit']}",
                "wind": f"{period['windSpeed']} {period['windDirection']}",
                "forecast": period["detailedForecast"],
            }
            for period in periods
        ]

        return forecast

    except requests.exceptions.RequestException as e:
        print(f"Request error fetching weather data: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error parsing weather data: {e}")
        return None
