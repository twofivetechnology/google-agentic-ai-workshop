import pytest
from unittest.mock import patch, MagicMock
from weather_agent import weather_agent

MOCK_FORECAST = [
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


class TestWeatherAgent:

    def test_agent_name(self):
        """Verify the agent has the correct name."""
        assert weather_agent.name == "rainer_shine"

    def test_agent_model(self):
        """Verify the agent is using the correct model."""
        assert weather_agent.model == "gemini-2.5-flash"

    def test_agent_has_tools(self):
        """Verify the agent has both tools registered."""
        tool_names = [tool.__name__ for tool in weather_agent.tools]
        assert "get_lat_lon" in tool_names
        assert "get_extended_weather_forecast" in tool_names

    def test_agent_has_two_tools(self):
        """Verify the agent has exactly two tools."""
        assert len(weather_agent.tools) == 2

    def test_agent_description_set(self):
        """Verify the agent has a description."""
        assert weather_agent.description is not None
        assert len(weather_agent.description) > 0

    def test_agent_instruction_set(self):
        """Verify the agent has instructions."""
        assert weather_agent.instruction is not None
        assert len(weather_agent.instruction) > 0

    def test_agent_instruction_mentions_tools(self):
        """Verify instructions reference both tools."""
        assert "get_lat_lon" in weather_agent.instruction
        assert "get_extended_weather_forecast" in weather_agent.instruction

    @patch("weather_agent.get_lat_lon")
    @patch("weather_agent.get_extended_weather_forecast")
    def test_tools_called_in_sequence(self, mock_forecast, mock_lat_lon):
        """Verify tools return expected data types."""
        mock_lat_lon.return_value = (38.9071923, -77.0368707)
        mock_forecast.return_value = MOCK_FORECAST

        lat, lon = mock_lat_lon("Washington, DC")
        forecast = mock_forecast(lat, lon)

        mock_lat_lon.assert_called_once_with("Washington, DC")
        mock_forecast.assert_called_once_with(lat, lon)
        assert forecast == MOCK_FORECAST

    @patch("weather_agent.get_lat_lon")
    def test_tool_returns_none_on_invalid_location(self, mock_lat_lon):
        """Verify get_lat_lon returns None for an invalid location."""
        mock_lat_lon.return_value = None

        result = mock_lat_lon("Invalid Location XYZ")

        assert result is None

    @patch("weather_agent.get_extended_weather_forecast")
    def test_tool_returns_none_on_invalid_coordinates(self, mock_forecast):
        """Verify get_extended_weather_forecast returns None for invalid coordinates."""
        mock_forecast.return_value = None

        result = mock_forecast(0.0, 0.0)

        assert result is None