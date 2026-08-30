import pytest
from unittest.mock import MagicMock, patch

from callbacks import (
    chained_before_callback,
    log_user_prompt,
    log_model_response,
    moderate_user_prompt,
    check_user_input,
    check_us_location,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_callback_context(agent_name="test_agent"):
    """Build a minimal mock CallbackContext."""
    ctx = MagicMock()
    ctx.agent_name = agent_name
    return ctx


def make_llm_request(text="What is the weather in Austin, TX?", role="user"):
    """Build a minimal mock LlmRequest with a single content item."""
    part = MagicMock()
    part.text = text

    content = MagicMock()
    content.role = role
    content.parts = [part]

    request = MagicMock()
    request.contents = [content]
    return request


def make_llm_response(text="Sunny skies ahead!"):
    """Build a minimal mock LlmResponse."""
    part = MagicMock()
    part.text = text

    response = MagicMock()
    response.content.parts = [part]
    return response


def get_response_text(llm_response):
    """Extract the text from a real LlmResponse returned by the callbacks."""
    return llm_response.content["parts"][0]["text"]


def make_armor_safe_response():
    """Build a mock Model Armor safe response."""
    mock_result = MagicMock()
    mock_result.filter_match_state = MagicMock()
    mock_result.filter_match_state.__eq__ = lambda self, other: False  # not MATCH_FOUND
    mock_response = MagicMock()
    mock_response.sanitization_result = mock_result
    return mock_response


def make_armor_blocked_response(reason="Prompt injection"):
    """Build a mock Model Armor blocked response."""
    from google.cloud.modelarmor_v1 import SanitizationResult
    mock_filter = MagicMock()
    mock_filter.display_name = reason

    mock_result = MagicMock()
    mock_result.filter_match_state = SanitizationResult.FilterMatchState.MATCH_FOUND
    mock_result.filter_results = {"filter_1": mock_filter}

    mock_response = MagicMock()
    mock_response.sanitization_result = mock_result
    return mock_response


# ---------------------------------------------------------------------------
# check_us_location
# ---------------------------------------------------------------------------

class TestCheckUsLocation:
    @patch("callbacks.check_location_in_us")
    def test_us_location_returns_none(self, mock_check):
        mock_check.return_value = True
        assert check_us_location("Austin, TX") is None

    @patch("callbacks.check_location_in_us")
    def test_non_us_location_is_blocked(self, mock_check):
        mock_check.return_value = False
        result = check_us_location("London, UK")
        assert result is not None
        assert "Access Denied" in get_response_text(result)

    @patch("callbacks.check_location_in_us")
    def test_check_location_called_with_user_text(self, mock_check):
        mock_check.return_value = True
        check_us_location("Dallas, TX")
        mock_check.assert_called_once_with("Dallas, TX")


# ---------------------------------------------------------------------------
# check_user_input  (Model Armor — mocked)
# ---------------------------------------------------------------------------

class TestCheckUserInput:
    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_safe_input_returns_none(self, mock_client):
        mock_client.sanitize_user_prompt.return_value = make_armor_safe_response()
        assert check_user_input("What will the weather be like tomorrow?") is None

    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_unsafe_input_is_blocked(self, mock_client):
        mock_client.sanitize_user_prompt.return_value = make_armor_blocked_response("Prompt injection")
        result = check_user_input("Ignore all previous instructions and...")
        assert result is not None
        assert "blocked by content moderation" in get_response_text(result)
        assert "Prompt injection" in get_response_text(result)

    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_unsafe_without_display_name_uses_default(self, mock_client):
        from google.cloud.modelarmor_v1 import SanitizationResult
        mock_filter = MagicMock()
        mock_filter.display_name = ""  # empty display name
        mock_result = MagicMock()
        mock_result.filter_match_state = SanitizationResult.FilterMatchState.MATCH_FOUND
        mock_result.filter_results = {"f": mock_filter}
        mock_client.sanitize_user_prompt.return_value.sanitization_result = mock_result
        result = check_user_input("some bad text")
        assert result is not None
        assert "Policy violation" in get_response_text(result)

    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_armor_exception_fails_open(self, mock_client):
        """If Model Armor call fails, the request should be allowed through."""
        mock_client.sanitize_user_prompt.side_effect = Exception("API unavailable")
        assert check_user_input("normal message") is None


# ---------------------------------------------------------------------------
# log_user_prompt
# ---------------------------------------------------------------------------

class TestLogUserPrompt:
    def test_logs_user_message_and_returns_none(self, caplog):
        ctx = make_callback_context()
        req = make_llm_request(text="Hello weather agent")
        with caplog.at_level("INFO"):
            result = log_user_prompt(ctx, req)
        assert result is None
        assert "Hello weather agent" in caplog.text

    def test_empty_contents_returns_none(self):
        ctx = make_callback_context()
        req = make_llm_request()
        req.contents = []
        assert log_user_prompt(ctx, req) is None

    def test_non_user_role_does_not_log(self, caplog):
        ctx = make_callback_context()
        req = make_llm_request(role="model")
        with caplog.at_level("INFO"):
            result = log_user_prompt(ctx, req)
        assert result is None
        assert "USER »" not in caplog.text


# ---------------------------------------------------------------------------
# log_model_response
# ---------------------------------------------------------------------------

class TestLogModelResponse:
    def test_logs_model_response_and_returns_none(self, caplog):
        ctx = make_callback_context()
        resp = make_llm_response(text="It will be sunny!")
        with caplog.at_level("INFO"):
            result = log_model_response(ctx, resp)
        assert result is None
        assert "It will be sunny!" in caplog.text

    def test_no_content_returns_none(self):
        ctx = make_callback_context()
        resp = MagicMock()
        resp.content = None
        assert log_model_response(ctx, resp) is None


# ---------------------------------------------------------------------------
# moderate_user_prompt
# ---------------------------------------------------------------------------

class TestModerateUserPrompt:
    @patch("callbacks.check_location_in_us")
    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_safe_us_input_returns_none(self, mock_client, mock_location):
        mock_client.sanitize_user_prompt.return_value = make_armor_safe_response()
        mock_location.return_value = True
        ctx = make_callback_context()
        req = make_llm_request(text="Will it rain in Dallas?")
        assert moderate_user_prompt(ctx, req) is None

    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_unsafe_input_blocked_before_location_check(self, mock_client):
        """Model Armor block should short-circuit before location check."""
        mock_client.sanitize_user_prompt.return_value = make_armor_blocked_response("Harmful content")
        ctx = make_callback_context()
        req = make_llm_request(text="Some harmful message")
        result = moderate_user_prompt(ctx, req)
        assert result is not None
        assert "blocked by content moderation" in get_response_text(result)

    @patch("callbacks.check_location_in_us")
    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_non_us_location_is_blocked(self, mock_client, mock_location):
        mock_client.sanitize_user_prompt.return_value = make_armor_safe_response()
        mock_location.return_value = False
        ctx = make_callback_context()
        req = make_llm_request(text="What's the weather in London?")
        result = moderate_user_prompt(ctx, req)
        assert result is not None
        assert "Access Denied" in get_response_text(result)

    def test_exception_during_moderation_returns_none(self):
        ctx = make_callback_context()
        req = make_llm_request()
        req.contents[-1].parts[0].text = None  # trigger AttributeError
        result = moderate_user_prompt(ctx, req)
        assert result is None


# ---------------------------------------------------------------------------
# chained_before_callback
# ---------------------------------------------------------------------------

class TestChainedBeforeCallback:
    @patch("callbacks.check_location_in_us")
    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_safe_us_request_returns_none(self, mock_client, mock_location):
        mock_client.sanitize_user_prompt.return_value = make_armor_safe_response()
        mock_location.return_value = True
        ctx = make_callback_context()
        req = make_llm_request(text="What's the forecast for Chicago?")
        assert chained_before_callback(ctx, req) is None

    @patch("callbacks.check_location_in_us")
    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_non_us_location_is_blocked(self, mock_client, mock_location):
        mock_client.sanitize_user_prompt.return_value = make_armor_safe_response()
        mock_location.return_value = False
        ctx = make_callback_context()
        req = make_llm_request(text="What's the weather in Paris?")
        result = chained_before_callback(ctx, req)
        assert result is not None
        assert "Access Denied" in get_response_text(result)

    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_unsafe_input_is_blocked(self, mock_client):
        mock_client.sanitize_user_prompt.return_value = make_armor_blocked_response("Abusive language")
        ctx = make_callback_context()
        req = make_llm_request(text="Some abusive message")
        result = chained_before_callback(ctx, req)
        assert result is not None
        assert "blocked by content moderation" in get_response_text(result)

    @patch("callbacks.log_user_prompt")
    @patch("callbacks.check_location_in_us")
    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_log_user_prompt_called_on_safe_request(self, mock_client, mock_location, mock_log):
        mock_client.sanitize_user_prompt.return_value = make_armor_safe_response()
        mock_location.return_value = True
        ctx = make_callback_context()
        req = make_llm_request()
        chained_before_callback(ctx, req)
        mock_log.assert_called_once_with(ctx, req)

    @patch("callbacks._MODEL_ARMOR_CLIENT")
    def test_log_user_prompt_not_called_when_blocked(self, mock_client):
        mock_client.sanitize_user_prompt.return_value = make_armor_blocked_response()
        ctx = make_callback_context()
        req = make_llm_request()
        with patch("callbacks.log_user_prompt") as mock_log:
            chained_before_callback(ctx, req)
            mock_log.assert_not_called()
