import pytest
from unittest.mock import MagicMock, patch
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

from critique_agent import critique_agent
from refine_agent import refine_agent
from search_agent import google_search_agent
from main_agent import main_agent, refined_response_team, append_to_state


# ---------------------------------------------------------------------------
# critique_agent tests
# ---------------------------------------------------------------------------

class TestCritiqueAgentConfiguration:
    def test_agent_is_llm_agent(self):
        assert isinstance(critique_agent, LlmAgent)

    def test_agent_name(self):
        assert critique_agent.name == "critique_agent"

    def test_agent_model(self):
        assert critique_agent.model == "gemini-2.5-flash"

    def test_agent_description_is_set(self):
        assert critique_agent.description
        assert len(critique_agent.description) > 0

    def test_agent_has_no_tools(self):
        assert len(critique_agent.tools) == 0

    def test_instruction_covers_accuracy(self):
        assert "accuracy" in critique_agent.instruction.lower()

    def test_instruction_covers_completeness(self):
        assert "completeness" in critique_agent.instruction.lower()

    def test_instruction_covers_clarity(self):
        assert "clarity" in critique_agent.instruction.lower()

    def test_instruction_defines_no_critique_sentinel(self):
        """Instruction must define NO_CRITIQUE_NEEDED so the pipeline can short-circuit."""
        assert "NO_CRITIQUE_NEEDED" in critique_agent.instruction

    def test_instruction_says_do_not_rewrite(self):
        """Critique agent must not rewrite — only critique."""
        assert "do not rewrite" in critique_agent.instruction.lower()


# ---------------------------------------------------------------------------
# refine_agent tests
# ---------------------------------------------------------------------------

class TestRefineAgentConfiguration:
    def test_agent_is_llm_agent(self):
        assert isinstance(refine_agent, LlmAgent)

    def test_agent_name(self):
        assert refine_agent.name == "refine_agent"

    def test_agent_model(self):
        assert refine_agent.model == "gemini-2.5-flash"

    def test_agent_description_is_set(self):
        assert refine_agent.description
        assert len(refine_agent.description) > 0

    def test_agent_has_no_tools(self):
        assert len(refine_agent.tools) == 0

    def test_instruction_references_critique(self):
        assert "critique" in refine_agent.instruction.lower()

    def test_instruction_references_original_response(self):
        assert "original_response" in refine_agent.instruction.lower()

    def test_instruction_says_no_meta_commentary(self):
        """Refine agent should output only the improved response, no preamble."""
        assert "meta-commentary" in refine_agent.instruction.lower()

    def test_instruction_says_preserve_good_parts(self):
        assert "preserve" in refine_agent.instruction.lower()

    def test_instruction_says_no_new_facts(self):
        assert "do not introduce" in refine_agent.instruction.lower()


# ---------------------------------------------------------------------------
# search_agent tests
# ---------------------------------------------------------------------------

class TestSearchAgentConfiguration:
    def test_agent_is_llm_agent(self):
        assert isinstance(google_search_agent, LlmAgent)

    def test_agent_name(self):
        assert google_search_agent.name == "google_search_agent"

    def test_agent_model(self):
        assert google_search_agent.model == "gemini-2.5-flash"

    def test_agent_description_is_set(self):
        assert google_search_agent.description
        assert "search" in google_search_agent.description.lower()

    def test_agent_has_google_search_tool(self):
        assert len(google_search_agent.tools) == 1
        assert google_search_agent.tools[0] == google_search

    def test_instruction_references_google_search_tool(self):
        assert "google_search" in google_search_agent.instruction.lower()

    def test_instruction_covers_honesty(self):
        instruction = google_search_agent.instruction.lower()
        assert any(word in instruction for word in ["honest", "cannot", "reliable"])

    def test_instruction_covers_authoritative_sources(self):
        assert "authoritative" in google_search_agent.instruction.lower()


# ---------------------------------------------------------------------------
# refined_response_team tests
# ---------------------------------------------------------------------------

class TestRefinedResponseTeam:
    def test_is_sequential_agent(self):
        assert isinstance(refined_response_team, SequentialAgent)

    def test_name(self):
        assert refined_response_team.name == "refined_response_team"

    def test_description_is_set(self):
        assert refined_response_team.description
        assert len(refined_response_team.description) > 0

    def test_has_three_sub_agents(self):
        assert len(refined_response_team.sub_agents) == 3

    def test_first_sub_agent_is_search(self):
        assert refined_response_team.sub_agents[0] == google_search_agent

    def test_second_sub_agent_is_critique(self):
        assert refined_response_team.sub_agents[1] == critique_agent

    def test_third_sub_agent_is_refine(self):
        assert refined_response_team.sub_agents[2] == refine_agent

    def test_pipeline_order_is_search_critique_refine(self):
        names = [a.name for a in refined_response_team.sub_agents]
        assert names == ["google_search_agent", "critique_agent", "refine_agent"]


# ---------------------------------------------------------------------------
# append_to_state tests
# ---------------------------------------------------------------------------

class TestAppendToState:
    def make_tool_context(self, initial_state=None):
        ctx = MagicMock()
        ctx.state = initial_state or {}
        return ctx

    def test_appends_to_empty_field(self):
        ctx = self.make_tool_context()
        result = append_to_state(ctx, "responses", "first response")
        assert ctx.state["responses"] == ["first response"]
        assert result == {"status": "success"}

    def test_appends_to_existing_field(self):
        ctx = self.make_tool_context({"responses": ["first response"]})
        append_to_state(ctx, "responses", "second response")
        assert ctx.state["responses"] == ["first response", "second response"]

    def test_does_not_overwrite_other_fields(self):
        ctx = self.make_tool_context({"other_field": "keep me"})
        append_to_state(ctx, "responses", "new response")
        assert ctx.state["other_field"] == "keep me"

    def test_returns_success_status(self):
        ctx = self.make_tool_context()
        result = append_to_state(ctx, "responses", "response")
        assert result["status"] == "success"

    def test_different_field_names(self):
        ctx = self.make_tool_context()
        append_to_state(ctx, "weather_responses", "sunny")
        append_to_state(ctx, "search_responses", "result")
        assert ctx.state["weather_responses"] == ["sunny"]
        assert ctx.state["search_responses"] == ["result"]


# ---------------------------------------------------------------------------
# main_agent tests
# ---------------------------------------------------------------------------

class TestMainAgentConfiguration:
    def test_agent_is_llm_agent(self):
        assert isinstance(main_agent, LlmAgent)

    def test_agent_name(self):
        assert main_agent.name == "main_agent"

    def test_agent_model(self):
        assert main_agent.model == "gemini-2.5-flash"

    def test_agent_description_is_set(self):
        assert main_agent.description
        assert len(main_agent.description) > 0

    def test_agent_instruction_is_set(self):
        assert main_agent.instruction
        assert len(main_agent.instruction) > 0


class TestMainAgentInstruction:
    def test_instruction_references_refined_response_team(self):
        assert "refined_response_team" in main_agent.instruction.lower()

    def test_instruction_references_append_to_state(self):
        assert "append_to_state" in main_agent.instruction.lower()

    def test_instruction_handles_no_critique_needed(self):
        assert "NO_CRITIQUE_NEEDED" in main_agent.instruction

    def test_instruction_covers_transparency(self):
        instruction = main_agent.instruction.lower()
        assert any(word in instruction for word in ["transparent", "unable", "fails"])


class TestMainAgentTools:
    def test_has_one_tool(self):
        assert len(main_agent.tools) == 1

    def test_tool_is_append_to_state(self):
        assert main_agent.tools[0] == append_to_state


class TestMainAgentSubAgents:
    def test_has_one_sub_agent(self):
        assert len(main_agent.sub_agents) == 1

    def test_sub_agent_is_refined_response_team(self):
        assert main_agent.sub_agents[0] == refined_response_team