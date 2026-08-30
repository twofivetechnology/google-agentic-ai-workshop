import pytest
from unittest.mock import patch, MagicMock
from location_tool import get_lat_lon, check_location_in_us, is_inside_us

MOCK_GEOCODE_RESPONSE = [
    {
        "geometry": {
            "location": {
                "lat": 38.9071923,
                "lng": -77.0368707
            }
        },
        "address_components": [
            {"types": ["country", "political"], "short_name": "US", "long_name": "United States"}
        ]
    }
]


@patch("location_tool.gmaps")
class TestGetLatLon:

    def test_successful_location(self, mock_gmaps):
        """Returns lat/lon tuple for a valid location."""
        mock_gmaps.geocode.return_value = MOCK_GEOCODE_RESPONSE

        result = get_lat_lon("Washington, DC")

        assert result == (38.9071923, -77.0368707)

    def test_returns_none_on_empty_result(self, mock_gmaps):
        """Returns None when geocoder returns no results."""
        mock_gmaps.geocode.return_value = []

        result = get_lat_lon("Nonexistent Place XYZ")

        assert result is None

    def test_returns_none_on_request_exception(self, mock_gmaps):
        """Returns None when a network error occurs."""
        import requests
        mock_gmaps.geocode.side_effect = requests.exceptions.RequestException("Network error")

        result = get_lat_lon("Washington, DC")

        assert result is None

    def test_returns_none_on_missing_geometry(self, mock_gmaps):
        """Returns None when geocode result is missing geometry keys."""
        mock_gmaps.geocode.return_value = [{"address_components": []}]

        result = get_lat_lon("Washington, DC")

        assert result is None

    def test_returns_none_on_exception(self, mock_gmaps):
        """Returns None when an unexpected exception occurs."""
        mock_gmaps.geocode.side_effect = Exception("Unexpected error")

        result = get_lat_lon("Washington, DC")

        assert result is None

    def test_geocode_called_with_location(self, mock_gmaps):
        """Verifies gmaps.geocode is called with the provided location string."""
        mock_gmaps.geocode.return_value = MOCK_GEOCODE_RESPONSE

        get_lat_lon("Washington, DC")

        mock_gmaps.geocode.assert_called_once_with("Washington, DC")


# ---------------------------------------------------------------------------
# Fixtures / mock data for check_location_in_us
# ---------------------------------------------------------------------------

def make_geocode_result(short_name="US", long_name="United States", include_country=True):
    """Build a minimal geocode result with address_components."""
    components = []
    if include_country:
        components.append({
            "types": ["country", "political"],
            "short_name": short_name,
            "long_name": long_name,
        })
    components.append({
        "types": ["locality", "political"],
        "short_name": "Austin",
        "long_name": "Austin",
    })
    return [{"address_components": components}]


# ---------------------------------------------------------------------------
# is_inside_us
# ---------------------------------------------------------------------------

class TestIsInsideUs:
    def test_us_short_name_returns_true(self):
        components = [{"types": ["country"], "short_name": "US", "long_name": "United States"}]
        assert is_inside_us(components) is True

    def test_us_long_name_returns_true(self):
        components = [{"types": ["country"], "short_name": "", "long_name": "United States"}]
        assert is_inside_us(components) is True

    def test_non_us_country_returns_false(self):
        components = [{"types": ["country"], "short_name": "GB", "long_name": "United Kingdom"}]
        assert is_inside_us(components) is False

    def test_no_country_component_returns_false(self):
        components = [{"types": ["locality"], "short_name": "Austin", "long_name": "Austin"}]
        assert is_inside_us(components) is False

    def test_empty_components_returns_false(self):
        assert is_inside_us([]) is False


# ---------------------------------------------------------------------------
# check_location_in_us
# ---------------------------------------------------------------------------

@patch("location_tool.gmaps")
class TestCheckLocationInUs:
    def test_us_location_returns_true(self, mock_gmaps):
        """A location in the US should return True."""
        mock_gmaps.geocode.return_value = make_geocode_result(short_name="US")
        assert check_location_in_us("Austin, TX") is True

    def test_non_us_location_returns_false(self, mock_gmaps):
        """A location outside the US should return False."""
        mock_gmaps.geocode.return_value = make_geocode_result(
            short_name="GB", long_name="United Kingdom"
        )
        assert check_location_in_us("London, UK") is False

    def test_no_results_returns_false(self, mock_gmaps):
        """Empty geocode result should return False."""
        mock_gmaps.geocode.return_value = []
        assert check_location_in_us("Nowhere XYZ") is False

    def test_no_country_component_returns_false(self, mock_gmaps):
        """Result with no country address component should return False."""
        mock_gmaps.geocode.return_value = make_geocode_result(include_country=False)
        assert check_location_in_us("Some Place") is False

    def test_geocode_exception_returns_false(self, mock_gmaps):
        """Exceptions from the geocoder should be caught and return False."""
        mock_gmaps.geocode.side_effect = Exception("API error")
        assert check_location_in_us("Austin, TX") is False

    def test_geocode_called_with_location(self, mock_gmaps):
        """Verifies gmaps.geocode is called with the provided location string."""
        mock_gmaps.geocode.return_value = make_geocode_result()
        check_location_in_us("Austin, TX")
        mock_gmaps.geocode.assert_called_once_with("Austin, TX")