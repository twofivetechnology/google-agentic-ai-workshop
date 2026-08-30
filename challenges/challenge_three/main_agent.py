from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from weather_agent import weather_agent
from search_agent import google_search_agent
from callbacks import chained_before_callback, log_model_response

main_agent = LlmAgent(
    name="main_agent",
    model="gemini-2.5-flash",
    description="Provides Answers to Users Questions.",
    instruction=(
        """You are a helpful assistant that answers user questions using the best available tool.

        You have access to the following tools and sub-agents:
        - **weather_agent**: Use this for any questions about weather forecasts, conditions,
          or climate for a specific location. Always delegate weather-related questions to this agent.
        - **google_search_agent**: Use this for general knowledge questions, current events,
          facts, or any topic that is not weather-related.

        When responding:
        1. Identify the intent of the user's question.
        2. If the question is about weather, delegate to the weather_agent.
        3. If the question requires up-to-date or factual information, use the google_search_agent tool.
        4. Synthesise the response from the sub-agent or tool into a clear, concise answer.
        5. If a question spans both weather and general knowledge, address each part with the appropriate tool.
        6. If you are unable to answer, be transparent and let the user know."""
    ),
    tools=[agent_tool.AgentTool(google_search_agent)],
    sub_agents=[weather_agent],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)