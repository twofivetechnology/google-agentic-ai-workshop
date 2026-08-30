from dotenv import load_dotenv
from location_tool import get_lat_lon
from weather_tool import get_extended_weather_forecast

load_dotenv()

def display_forecast(forecast):
    """Display the forecast in a readable format."""
    for period in forecast:
        print(f"\n📅 {period['name']}")
        print(f"🌡️  Temperature: {period['temperature']}")
        print(f"💨 Wind: {period['wind']}")
        print(f"🌤️  Forecast: {period['forecast']}")
        print("-" * 40)


def main():
    print("🌦️  Welcome to Rainer Shine's Weather Bot!")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        location = input("Enter a location (e.g., 'Washington, DC'): ").strip()

        if location.lower() in ("quit", "exit"):
            print("Thanks for using Rainer Shine! Stay dry out there! ☂️")
            break

        if not location:
            print("Please enter a valid location.\n")
            continue

        print(f"\n🔍 Looking up coordinates for '{location}'...")
        coords = get_lat_lon(location)

        if not coords:
            print(f"❌ Could not find coordinates for '{location}'. Please try again.\n")
            continue

        lat, lon = coords
        print(f"📍 Found: ({lat}, {lon})")
        print(f"⛅ Fetching forecast...\n")

        forecast = get_extended_weather_forecast(lat, lon)

        if not forecast:
            print(f"❌ Could not retrieve forecast for '{location}'. Please try again.\n")
            continue

        print(f"🌈 Extended Forecast for {location}:")
        print("=" * 40)
        display_forecast(forecast)
        print()


if __name__ == "__main__":
    main()
