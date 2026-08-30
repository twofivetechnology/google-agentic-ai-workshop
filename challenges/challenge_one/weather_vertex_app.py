import json
from vertexai.preview import reasoning_engines
from weather_agent import weather_agent

app = reasoning_engines.AdkApp(
    agent=weather_agent,
    enable_tracing=True,
)


def extract_text(chunk) -> str:
    """Extract only text parts from a streaming chunk."""
    try:
        parts = chunk.get("content", {}).get("parts", [])
        return "".join(
            part["text"]
            for part in parts
            if "text" in part
        )
    except (AttributeError, KeyError):
        return ""

def main():
    print("🌦️  Welcome to Rainer Shine's Weather Bot on Vertex AI!")
    print("Type 'goodbye' to exit.\n")

    user_id = "test-user-id"
    session = app.create_session(user_id=user_id)

    while True:
        location = input("Enter a location (e.g., 'Washington, DC'): ").strip()

        if location.lower() == "goodbye":
            print("Goodbye! Stay sunny side up! ☀️")
            break

        if not location:
            print("Please enter a valid location.\n")
            continue

        print(f"\n⛅ Fetching forecast for '{location}'...\n")

        try:
            raw_result = ""
            for chunk in app.stream_query(
                user_id=user_id,
                message=f"What is the weather forecast for {location}?",
            ):
                text = extract_text(chunk)
                if text:
                    print(text, end="", flush=True)
            print("\n")

        except Exception as e:
            print(f"❌ Error fetching forecast: {e}\n")


if __name__ == "__main__":
    main()
