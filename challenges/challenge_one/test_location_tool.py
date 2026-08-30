import pytest
from unittest.mock import patch, MagicMock
from location_tool import get_lat_lon

MOCK_GEOCODE_RESPONSE = {
    "status": "OK",
    "results": [
        {
            "geometry": {
                "location": {
                    "lat": 38.9071923,
                    "lng": -77.0368707
                }
            }
        }
    ]
}


def make_mock_response(json_data, status_code=200):
    """Helper to create a mock requests response."""
    mock_response = MagicMock()
    mock_response.json.return_value = json_data
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()
    return mock_response


@patch("location_tool.requests.get")
@patch("location_tool.os.getenv")
class TestGetLatLon:

    def test_successful_location(self, mock_getenv, mock_get):
        """Returns lat/lon tuple for a valid location."""
        mock_getenv.return_value = "fake-api-key"
        mock_get.return_value = make_mock_response(MOCK_GEOCODE_RESPONSE)

        result = get_lat_lon("Washington, DC")

        assert result == (38.9071923, -77.0368707)

    def test_returns_none_when_no_api_key(self, mock_getenv, mock_get):
        """Returns None when API key is not set."""
        mock_getenv.return_value = None

        result = get_lat_lon("Washington, DC")

        assert result is None
        mock_get.assert_not_called()

    def test_returns_none_on_bad_status(self, mock_getenv, mock_get):
        """Returns None when Geocoding API returns a non-OK status."""
        mock_getenv.return_value = "fake-api-key"
        mock_get.return_value = make_mock_response({"status": "ZERO_RESULTS", "results": []})

        result = get_lat_lon("Nonexistent Place XYZ")

        assert result is None

    def test_returns_none_on_request_exception(self, mock_getenv, mock_get):
        """Returns None when a network error occurs."""
        import requests
        mock_getenv.return_value = "fake-api-key"
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = get_lat_lon("Washington, DC")

        assert result is None

    def test_returns_none_on_missing_key(self, mock_getenv, mock_get):
        """Returns None when API response is missing expected keys."""
        mock_getenv.return_value = "fake-api-key"
        mock_get.return_value = make_mock_response({"status": "OK", "results": []})

        result = get_lat_lon("Washington, DC")

        assert result is None

    def test_correct_api_url_called(self, mock_getenv, mock_get):
        """Verifies the correct URL and params are used."""
        mock_getenv.return_value = "fake-api-key"
        mock_get.return_value = make_mock_response(MOCK_GEOCODE_RESPONSE)

        get_lat_lon("Washington, DC")

        mock_get.assert_called_once_with(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": "Washington, DC", "key": "fake-api-key"}
        )