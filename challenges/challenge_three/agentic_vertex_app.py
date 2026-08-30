import json
from vertexai.preview import reasoning_engines
from main_agent import main_agent

app = reasoning_engines.AdkApp(
    agent=main_agent,
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
    print("Welcome! Ask anything or type 'goodbye' to exit.\n")

    user_id = "test-user-id"
    session = app.create_session(user_id=user_id)

    while True:
        user_input = input("How can I help?").strip()

        if user_input.lower() == "goodbye":
            print("Goodbye! Stay sunny side up! ☀️")
            break

        if not user_input:
            print("Please enter how I can help.\n")
            continue

        try:
            raw_result = ""
            for chunk in app.stream_query(
                user_id=user_id,
                message=user_input,
            ):
                text = extract_text(chunk)
                if text:
                    print(text, end="", flush=True)
            print("\n")

        except Exception as e:
            print(f"❌ Error fetching forecast: {e}\n")


if __name__ == "__main__":
    main()
