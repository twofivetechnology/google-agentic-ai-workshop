import json
from vertexai.preview import reasoning_engines
import vertexai
from vertexai import agent_engines
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import agent_tool, google_search
from critique_agent import critique_agent
from refine_agent import refine_agent
import logging
import os
from typing import Optional
from google import genai
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.cloud import modelarmor_v1
import googlemaps
from typing import Optional, Tuple
import requests
from typing import Optional, List, Dict

def get_map_routes(
    lat: float,
    lon: float,
    mode: str = "driving",
) -> Optional[List[Dict[str, str]]]:
    """
    Find best evacuation routes away from the given lat/lon using Google Maps Directions API.

    Args:
        lat (float): Latitude of the origin location.
        lon (float): Longitude of the origin location.
        mode (str): Travel mode — 'driving', 'walking', 'bicycling', or 'transit'.

    Returns:
        Optional[List[Dict[str, str]]]: A list of route summaries, or None on error.
    """
    try:
        origin = (lat, lon)

        # Use the geocode reverse lookup to find the nearest major city to route towards
        reverse = gmaps.reverse_geocode(origin)
        if not reverse:
            print(f"Could not reverse geocode ({lat}, {lon})")
            return None

        # Extract state/region to find a destination inland away from the origin
        address_components = reverse[0].get("address_components", [])
        state = next(
            (c["long_name"] for c in address_components if "administrative_area_level_1" in c["types"]),
            None,
        )
        if not state:
            print("Could not determine state from coordinates.")
            return None

        # Route towards the state capital as a safe inland destination
        destination = f"{state} state capital"

        directions = gmaps.directions(
            origin=origin,
            destination=destination,
            mode=mode,
            alternatives=True,
        )

        if not directions:
            print(f"No routes found from ({lat}, {lon}) to {destination}")
            return None

        routes = []
        for i, route in enumerate(directions):
            leg = route["legs"][0]
            routes.append({
                "route_number": str(i + 1),
                "summary": route.get("summary", "No summary available"),
                "destination": leg["end_address"],
                "distance": leg["distance"]["text"],
                "duration": leg["duration"]["text"],
                "start_address": leg["start_address"],
                "end_address": leg["end_address"],
                "steps": " → ".join(
                    step["html_instructions"]
                    for step in leg.get("steps", [])
                ),
            })

        return routes

    except Exception as e:
        print(f"Error fetching map routes: {e}")
        return None

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

# Initialize the client with your Google Maps API key
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=API_KEY)

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

        geocode_result = gmaps.geocode(location)

        if not geocode_result:
            print(f"No results found for location: '{location}'")
            return None

        lat_lon = geocode_result[0].get("geometry").get("location")
        return (lat_lon["lat"], lat_lon["lng"])

    except Exception as e:
        print(f"Request error fetching location data: {e}")
        return None

def check_location_in_us(location: str) -> bool:
    """
    Checks if a written location string is inside the United States.
    """
    try:
        # Perform forward geocoding
        geocode_result = gmaps.geocode(location)
        
        if not geocode_result:
            print(f"No results found for location: '{location}'")
            return False
            
        # Parse the first matching result
        address_components = geocode_result[0].get("address_components", [])
        return is_inside_us(address_components)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def is_inside_us(address_components):
    """
    Helper function to parse Google Maps address components for the United States.
    """
    for component in address_components:
        types = component.get("types", [])
        # Look specifically for the country component
        if "country" in types:
            short_name = component.get("short_name", "")
            long_name = component.get("long_name", "")
            return short_name == "US" or long_name == "United States"
    return False

_MODEL_ARMOR_CLIENT = modelarmor_v1.ModelArmorClient(client_options={
        "api_endpoint": "modelarmor.us-central1.rep.googleapis.com"
    })
_MODEL_ARMOR_TEMPLATE = "projects/qwiklabs-gcp-03-18da3cda13cf/locations/us-central1/templates/challenge_two"

logger = logging.getLogger(__name__)

def chained_before_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    # 1. Moderation check
    moderation_result = moderate_user_prompt(callback_context, llm_request, False)
    if moderation_result is not None:
        return moderation_result  # STOP: message was inappropriate
    # 2. Log user input
    log_user_prompt(callback_context, llm_request)
    return None  # Allow agent to proceed

def chained_before_callback_with_location(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    # 1. Moderation check
    moderation_result = moderate_user_prompt(callback_context, llm_request, True)
    if moderation_result is not None:
        return moderation_result  # STOP: message was inappropriate
    # 2. Log user input
    log_user_prompt(callback_context, llm_request)
    return None  # Allow agent to proceed


def log_user_prompt(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    if llm_request.contents:
        first = llm_request.contents[0]
        if first.role == "user" and first.parts and first.parts[0].text:
            logger.info(
                "[%s] USER » %s",
                callback_context.agent_name,
                first.parts[0].text.strip(),
            )
    return None


def log_model_response(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    if llm_response.content and llm_response.content.parts:
        txt = llm_response.content.parts[0].text
        if txt:
            logger.info("[%s] MODEL » %s", callback_context.agent_name, txt.strip())
    return None


def moderate_user_prompt(
    callback_context: CallbackContext, llm_request: LlmRequest, check_location: bool
) -> Optional[LlmResponse]:
    try:
        first = llm_request.contents[0]
        if first.role == "user" and first.parts and first.parts[0].text:
            user_text = first.parts[0].text.strip()
            text_check = check_user_input(user_text)
            if text_check:
                return text_check
            if check_location:
                location_check = check_us_location(user_text)
                if location_check:
                    return location_check
    except Exception as e:
        logging.exception("Moderation callback failed: %s", e)
    return None

def check_user_input(user_text: str) -> Optional[LlmResponse]:
    """Uses Google Cloud Model Armor to evaluate whether the user's message is safe."""
    try:
        request = modelarmor_v1.SanitizeUserPromptRequest(
            name=_MODEL_ARMOR_TEMPLATE,
            user_prompt_data=modelarmor_v1.DataItem(text=user_text),
        )
        response = _MODEL_ARMOR_CLIENT.sanitize_user_prompt(request)
        result = response.sanitization_result

        if result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
            matched = [
                f.display_name
                for f in result.filter_results.values()
                if hasattr(f, "display_name") and f.display_name
            ]
            reason = ", ".join(matched) if matched else "Policy violation"
            logger.warning("[moderation] Model Armor blocked input: %s", reason)
            return LlmResponse(
                content={
                    "role": "model",
                    "parts": [{"text": f"Message blocked by content moderation: {reason}"}],
                }
            )

        logger.info("[moderation] Model Armor verdict: SAFE")

    except Exception as e:
        logger.exception("Model Armor moderation check failed: %s", e)

    return None  # SAFE — allow the request to proceed


def check_us_location(
    user_text: str,
) -> Optional[LlmResponse]:
    """
    Validates if the user's input is 'US'.
    Returns None to proceed normally, or an LlmResponse to block and skip the LLM.
    """
    if not check_location_in_us(user_text):
        return LlmResponse(
            content={
                "role": "model",
                "parts": [
                    {
                        "text": "Access Denied: This service is only available for locations in the United States."
                    }
                ],
            }
        )

    return None

weather_agent = Agent(
    name="rainer_shine",
    model="gemini-2.5-flash",
    description=(
        "Rainer Shine is a friendly weather agent that delivers the forecast with a wry wit and sense of humor."
    ),
    instruction=(
        """You are Rainer Shine, a friendly weather assistant with a wry wit and sense of humor.
        When a user asks for a weather forecast:
        1. Use the get_lat_lon tool to convert the location name to latitude and longitude.
        2. Use the get_extended_weather_forecast tool with the retrieved lat/lon to fetch the forecast.
        3. Present the forecast in a fun and engaging way with light weather puns and humor.
        If asked about non-weather topics, politely redirect the conversation back to weather."""
    ),
    tools=[get_lat_lon, get_extended_weather_forecast],
    before_model_callback=chained_before_callback_with_location,
    after_model_callback=log_model_response,
)

map_agent = Agent(
    name="map_agent",
    model="gemini-2.5-flash",
    description=(
        "Helpful agent for finding routes such as evacuation routes."
    ),
    instruction=(
        """You are a friendly map assistant with a wry wit and sense of humor.
        When a user asks for a mapping/routes:
        1. Use the get_lat_lon tool to convert the location name to latitude and longitude.
        2. Use the get_map_routes tool with the retrieved lat/lon to fetch the routes.
        3. Present the routes in a fun and engaging way with light traffic puns and humor.
        If asked about non-map topics, politely redirect the conversation back to weather."""
    ),
    tools=[get_lat_lon, get_map_routes],
    before_model_callback=chained_before_callback_with_location,
    after_model_callback=log_model_response,
)

google_search_agent = LlmAgent(
    name="google_search_agent",
    model="gemini-2.5-flash",
    description="Provides accurate, up-to-date information by searching the web using Google Search.",
    instruction=(
        """You are a research assistant with access to Google Search.

        When a user asks a question:
        1. Use the google_search tool to find accurate, up-to-date information.
        2. Summarise the results clearly and concisely, citing the key facts.
        3. If the search returns multiple relevant sources, synthesise them into a single coherent answer.
        4. If the question is ambiguous, clarify your interpretation before answering.
        5. Always prefer recent, authoritative sources.
        6. If you cannot find a reliable answer, say so honestly rather than guessing."""
    ),
    tools=[google_search],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

critique_agent = LlmAgent(
    name="critique_agent",
    model="gemini-2.5-flash",
    description="Takes in search results and provides critiques to improve them.",
    instruction=(
        """You are a critical review assistant. Your sole job is to analyse a response
        and provide clear, actionable feedback to improve it.

        When critiquing a response:
        1. Evaluate the response for accuracy — flag any factual errors or unsupported claims.
        2. Evaluate the response for completeness — identify any important points that are missing.
        3. Evaluate the response for clarity — highlight any ambiguous, confusing, or overly verbose sections.
        4. Evaluate the response for structure — suggest improvements to ordering or formatting if needed.
        5. Evaluate the response for tone — flag anything that is inappropriate, biased, or unhelpful.

        When writing your critique:
        - Be specific and constructive — reference the exact part of the response you are critiquing.
        - Be concise — list only genuine issues, not minor stylistic preferences.
        - Do not rewrite the response yourself — only provide the critique.
        - If the response is of high quality and requires no changes, respond with: NO_CRITIQUE_NEEDED"""
    ),
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

refine_agent = LlmAgent(
    name="refine_agent",
    model="gemini-2.5-flash",
    description="Takes in critiques and refines the response accordingly.",
    instruction=(
        """You are a refinement assistant. Your sole job is to improve a previous response
        based on a critique provided to you.

        You will receive:
        - **original_response**: The initial response that was generated.
        - **critique**: Specific feedback identifying weaknesses, inaccuracies, or areas for improvement.

        When refining:
        1. Read the critique carefully and understand every point raised.
        2. Revise the original response to directly address each critique point.
        3. Preserve any parts of the original response that were not criticised.
        4. Do not introduce new factual claims that are not supported by the original response.
        5. Ensure the refined response is clear, concise, and well-structured.
        6. Do not include meta-commentary such as 'Here is the refined response' —
           output only the improved response itself."""
    ),
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

multi_agent = LlmAgent(
    name="multi_agent",
    model="gemini-2.5-flash",
    description="Provides Answers to Users Questions.",
    instruction=(
        """You are a helpful assistant that answers user questions using the best available tool.

        You have access to the following tools and sub-agents:
        - **weather_agent**: Use this for any questions about weather forecasts, conditions,
          or climate for a specific location. Always delegate weather-related questions to this agent.
        - **map_agent**: Use this for any questions about map routes. Always delegate map-related questions to this agent.
        - **google_search_agent**: Use this for general knowledge questions, current events,
          facts, or any topic that is not weather-related.

        When responding:
        1. Identify the intent of the user's question.
        2. If the question is about weather, delegate to the weather_agent.
        3. If the question is about maps/routes, delegate to the map_agent.
        4. If the question requires up-to-date or factual information, use the google_search_agent tool.
        5. Synthesise the response from the sub-agent or tool into a clear, concise answer.
        6. If a question spans both weather and general knowledge, address each part with the appropriate tool.
        7. If you are unable to answer, be transparent and let the user know."""
    ),
    tools=[agent_tool.AgentTool(google_search_agent)],
    sub_agents=[weather_agent, map_agent],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

def append_to_state(tool_context, field, response):
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    return {"status": "success"}

refined_response_team = SequentialAgent(
    name="refined_response_team",
    description="Performs an action, critiques the response, then refines it.",
    sub_agents=[
        multi_agent,
        critique_agent,
        refine_agent,
    ],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

main_agent = LlmAgent(
    name="main_agent",
    model="gemini-2.5-flash",
    description="Provides Answers to Users Questions.",
    instruction=(
        """You are a helpful FEMA tool that can forecast weather, determine evacuation routes.

        You have access to the following:
        - **refined_response_team**: A sequential pipeline that determines an answer,
          critiques the result, and refines it into a high-quality response. Use this for
          general knowledge questions, current events, or any factual queries.
        - **append_to_state**: A tool to persist responses into session state for later use.
          Call this after receiving a final refined response to store it under an appropriate field name.

        When handling a user request:
        1. Identify the intent of the question.
        2. Delegate to the refined_response_team for general or factual questions.
        3. Once the refined_response_team returns its final answer, present it clearly to the user.
        4. Use append_to_state to save the final response to session state with a descriptive field name.
        5. If the refined_response_team returns NO_CRITIQUE_NEEDED at the critique stage,
           the original search response is already high quality — present it directly.
        6. If you are unable to answer or the pipeline fails, be transparent with the user."""
    ),
    tools=[append_to_state],
    sub_agents=[refined_response_team],
)

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
    requirements=["google-cloud-aiplatform[agent_engines,adk]",
                  "requests",
                  "googlemaps",
                  "google-adk",
                  "google-cloud-modelarmor",
                  "python-dotenv"],
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
    user_id = "test-user-id"
    session = app.create_session(user_id=user_id)

    try:
        for chunk in remote_agent.stream_query(
            user_id=user_id,
            message="what can this agent do?",
        ):
            print(chunk, end="", flush=True)
            text = extract_text(chunk)
            if text:
                print(text, end="", flush=True)
        print("\n")
        for chunk in remote_agent.stream_query(
            user_id=user_id,
            message="what is there to do this week in the Outer Banks, North Carolina?",
        ):
            print(chunk, end="", flush=True)
            text = extract_text(chunk)
            if text:
                print(text, end="", flush=True)
        print("\n")
        for chunk in remote_agent.stream_query(
            user_id=user_id,
            message="What does the weather look like tomorrow in the Outer Banks, North Carolina?",
        ):
            print(chunk, end="", flush=True)
            text = extract_text(chunk)
            if text:
                print(text, end="", flush=True)
        print("\n")
        for chunk in remote_agent.stream_query(
            user_id=user_id,
            message="If there is a hurricane in the Outer Banks, North Carolina, how should I get out of town?",
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
