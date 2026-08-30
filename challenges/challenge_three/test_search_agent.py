import pytest
from unittest.mock import MagicMock, patch
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from search_agent import google_search_agent


# ---------------------------------------------------------------------------
# Agent configuration tests
# ---------------------------------------------------------------------------

class TestSearchAgentConfiguration:
    def test_agent_is_llm_agent(self):
        """Agent should be an LlmAgent instance."""
        assert isinstance(google_search_agent, LlmAgent)

    def test_agent_name(self):
        """Agent should have the correct name."""
        assert google_search_agent.name == "google_search_agent"

    def test_agent_model(self):
        """Agent should use the correct model."""
        assert google_search_agent.model == "gemini-2.5-flash"

    def test_agent_description_is_set(self):
        """Agent description should be non-empty."""
        assert google_search_agent.description
        assert len(google_search_agent.description) > 0

    def test_agent_instruction_is_set(self):
        """Agent instruction should be non-empty."""
        assert google_search_agent.instruction
        assert len(google_search_agent.instruction) > 0

    def test_agent_has_google_search_tool(self):
        """Agent should have exactly one tool: google_search."""
        assert len(google_search_agent.tools) == 1
        assert google_search_agent.tools[0] == google_search

    def test_agent_instruction_covers_search_behaviour(self):
        """Instruction should reference key search behaviours."""
        instruction = google_search_agent.instruction.lower()
        assert "search" in instruction
        assert "google_search" in instruction

    def test_agent_instruction_covers_honesty(self):
        """Instruction should tell the agent to be honest when it can't find an answer."""
        instruction = google_search_agent.instruction.lower()
        assert any(word in instruction for word in ["honest", "cannot", "reliable"])

    def test_agent_description_mentions_search(self):
        """Description should mention search so sub-agent routing works correctly."""
        assert "search" in google_search_agent.description.lower()