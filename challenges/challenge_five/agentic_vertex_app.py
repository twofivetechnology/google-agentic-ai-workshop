import json
from vertexai.preview import reasoning_engines
from main_agent import main_agent
import vertexai
from vertexai import agent_engines

vertexai.init(
project="qwiklabs-gcp-03-18da3cda13cf",
location="us-central1",
staging_bucket="gs://vertex_staging_bucket_qwiklabs-gcp-03",
)

app = reasoning_engines.AdkApp(
    agent=main_agent,
    enable_tracing=True,
)

remote_agent = agent_engines.create(
    app,
    requirements=["google-cloud-aiplatform[agent_engines,adk]"],
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


    try:
        raw_result = ""
        for chunk in remote_agent.stream_query(
            user_id=user_id,
            message="Give me the news highlights in the world of sports.",
        ):
            print(chunk, end="", flush=True)
            text = extract_text(chunk)
            if text:
                print(text, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"❌ Error fetching forecast: {e}\n")


if __name__ == "__main__":
    main()
