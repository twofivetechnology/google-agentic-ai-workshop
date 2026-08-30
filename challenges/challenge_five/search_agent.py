from google.adk.agents import LlmAgent
from google.adk.tools import google_search

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
)