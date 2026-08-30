from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import agent_tool
from search_agent import google_search_agent
from critique_agent import critique_agent
from refine_agent import refine_agent


def append_to_state(tool_context, field, response):
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    return {"status": "success"}


refined_response_team = SequentialAgent(
    name="refined_response_team",
    description="Performs a search, critiques the response, then refines it.",
    sub_agents=[
        google_search_agent,
        critique_agent,
        refine_agent,
    ],
)

main_agent = LlmAgent(
    name="main_agent",
    model="gemini-2.5-flash",
    description="Provides Answers to Users Questions.",
    instruction=(
        """You are a helpful orchestration assistant that routes user questions to the
        most appropriate agent and delivers clear, refined answers.

        You have access to the following:
        - **refined_response_team**: A sequential pipeline that searches for information,
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