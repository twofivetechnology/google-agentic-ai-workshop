import pytest
from unittest.mock import patch, MagicMock
from weather_tool import get_extended_weather_forecast

# Sample mock API responses
MOCK_POINTS_RESPONSE = {
    "properties": {
        "forecast": "https://api.weather.gov/gridpoints/LWX/97,71/forecast"
    }
}

MOCK_FORECAST_RESPONSE = {
    "properties": {
        "periods": [
            {
                "name": "Tonight",
                "temperature": 55,
                "temperatureUnit": "F",
                "windSpeed": "10 mph",
                "windDirection": "NW",
                "detailedForecast": "Mostly clear skies with a light breeze.",
            },
            {
                "name": "Tuesday",
                "temperature": 72,
                "temperatureUnit": "F",
                "windSpeed": "15 mph",
                "windDirection": "SW",
                "detailedForecast": "Sunny with a high near 72.",
            },
        ]
    }
}

EXPECTED_FORECAST = [
    {
        "name": "Tonight",
        "temperature": "55°F",
        "wind": "10 mph NW",
        "forecast": "Mostly clear skies with a light breeze.",
    },
    {
        "name": "Tuesday",
        "temperature": "72°F",
        "wind": "15 mph SW",
        "forecast": "Sunny with a high near 72.",
    },
]


def make_mock_response(json_data, status_code=200):
    """Helper to create a mock requests response."""
    mock_response = MagicMock()
    mock_response.json.return_value = json_data
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()
    return mock_response


@patch("weather_tool.requests.get")
class TestGetExtendedWeatherForecast:

    def test_successful_forecast(self, mock_get):
        """Returns a parsed forecast list on a successful API call."""
        mock_get.side_effect = [
            make_mock_response(MOCK_POINTS_RESPONSE),
            make_mock_response(MOCK_FORECAST_RESPONSE),
        ]

        result = get_extended_weather_forecast(38.8977, -77.0365)

        assert result == EXPECTED_FORECAST
        assert mock_get.call_count == 2

    def test_returns_none_on_request_exception(self, mock_get):
        """Returns None when a network error occurs."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = get_extended_weather_forecast(38.8977, -77.0365)

        assert result is None

    def test_returns_none_on_missing_key(self, mock_get):
        """Returns None when API response is missing expected keys."""
        mock_get.side_effect = [
            make_mock_response({"properties": {}}),  # Missing 'forecast' key
        ]

        result = get_extended_weather_forecast(38.8977, -77.0365)

        assert result is None

    def test_returns_none_on_bad_forecast_data(self, mock_get):
        """Returns None when forecast periods are malformed."""
        mock_get.side_effect = [
            make_mock_response(MOCK_POINTS_RESPONSE),
            make_mock_response({"properties": {"periods": [{"bad_key": "data"}]}}),
        ]

        result = get_extended_weather_forecast(38.8977, -77.0365)

        assert result is None

    def test_correct_api_urls_called(self, mock_get):
        """Verifies the correct URLs are called with the given coordinates."""
        mock_get.side_effect = [
            make_mock_response(MOCK_POINTS_RESPONSE),
            make_mock_response(MOCK_FORECAST_RESPONSE),
        ]

        get_extended_weather_forecast(38.8977, -77.0365)

        calls = mock_get.call_args_list
        assert "38.8977,-77.0365" in calls[0][0][0]
        assert calls[1][0][0] == MOCK_POINTS_RESPONSE["properties"]["forecast"]

    def test_empty_periods_returns_empty_list(self, mock_get):
        """Returns an empty list when forecast has no periods."""
        mock_get.side_effect = [
            make_mock_response(MOCK_POINTS_RESPONSE),
            make_mock_response({"properties": {"periods": []}}),
        ]

        result = get_extended_weather_forecast(38.8977, -77.0365)

        assert result == []