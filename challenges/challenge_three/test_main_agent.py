import pytest
from unittest.mock import MagicMock, patch
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

from main_agent import main_agent
from weather_agent import weather_agent
from search_agent import google_search_agent


# ---------------------------------------------------------------------------
# Agent configuration tests
# ---------------------------------------------------------------------------

class TestMainAgentConfiguration:
    def test_agent_is_llm_agent(self):
        """main_agent should be an LlmAgent instance."""
        assert isinstance(main_agent, LlmAgent)

    def test_agent_name(self):
        """Agent should have the correct name."""
        assert main_agent.name == "main_agent"

    def test_agent_description_is_set(self):
        """Agent description should be non-empty."""
        assert main_agent.description
        assert len(main_agent.description) > 0

    def test_agent_instruction_is_set(self):
        """Agent instruction should be non-empty."""
        assert main_agent.instruction
        assert len(main_agent.instruction) > 0


# ---------------------------------------------------------------------------
# Instruction content tests
# ---------------------------------------------------------------------------

class TestMainAgentInstruction:
    def test_instruction_references_weather_agent(self):
        """Instruction should mention weather_agent for routing."""
        assert "weather_agent" in main_agent.instruction.lower()

    def test_instruction_references_search_agent(self):
        """Instruction should mention google_search_agent for routing."""
        assert "google_search_agent" in main_agent.instruction.lower()

    def test_instruction_covers_delegation(self):
        """Instruction should describe how to delegate to sub-agents."""
        instruction = main_agent.instruction.lower()
        assert any(word in instruction for word in ["delegate", "use", "route"])

    def test_instruction_covers_transparency(self):
        """Instruction should tell the agent to be transparent when it cannot answer."""
        instruction = main_agent.instruction.lower()
        assert any(word in instruction for word in ["unable", "transparent", "cannot", "let the user know"])


# ---------------------------------------------------------------------------
# Tools tests
# ---------------------------------------------------------------------------

class TestMainAgentTools:
    def test_agent_has_one_tool(self):
        """main_agent should have exactly one tool."""
        assert len(main_agent.tools) == 1

    def test_tool_is_agent_tool(self):
        """The tool should be an AgentTool wrapping google_search_agent."""
        tool = main_agent.tools[0]
        assert isinstance(tool, agent_tool.AgentTool)

    def test_agent_tool_wraps_search_agent(self):
        """The AgentTool should wrap google_search_agent."""
        tool = main_agent.tools[0]
        assert tool.agent == google_search_agent


# ---------------------------------------------------------------------------
# Sub-agents tests
# ---------------------------------------------------------------------------

class TestMainAgentSubAgents:
    def test_agent_has_one_sub_agent(self):
        """main_agent should have exactly one sub-agent."""
        assert len(main_agent.sub_agents) == 1

    def test_sub_agent_is_weather_agent(self):
        """The sub-agent should be the weather_agent."""
        assert main_agent.sub_agents[0] == weather_agent

    def test_weather_agent_name(self):
        """The weather sub-agent should have the correct name."""
        assert main_agent.sub_agents[0].name == "rainer_shine"


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------

class TestMainAgentCallbacks:
    def test_before_model_callback_is_set(self):
        """before_model_callback should be configured."""
        assert main_agent.before_model_callback is not None

    def test_after_model_callback_is_set(self):
        """after_model_callback should be configured."""
        assert main_agent.after_model_callback is not None

    def test_before_model_callback_is_chained(self):
        """before_model_callback should be chained_before_callback."""
        from callbacks import chained_before_callback
        assert main_agent.before_model_callback == chained_before_callback

    def test_after_model_callback_is_log_model_response(self):
        """after_model_callback should be log_model_response."""
        from callbacks import log_model_response
        assert main_agent.after_model_callback == log_model_response