import logging
import os
from typing import Optional

from google import genai
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from location_tool import check_location_in_us

from google.cloud import modelarmor_v1

_MODEL_ARMOR_CLIENT = modelarmor_v1.ModelArmorClient(client_options={
        "api_endpoint": "modelarmor.us-central1.rep.googleapis.com"
    })
_MODEL_ARMOR_TEMPLATE = "projects/qwiklabs-gcp-03-18da3cda13cf/locations/us-central1/templates/challenge_two"

logger = logging.getLogger(__name__)

def chained_before_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    # 1. Moderation check
    moderation_result = moderate_user_prompt(callback_context, llm_request)
    if moderation_result is not None:
        return moderation_result  # STOP: message was inappropriate
    # 2. Log user input
    log_user_prompt(callback_context, llm_request)
    return None  # Allow agent to proceed


def log_user_prompt(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    if llm_request.contents:
        last = llm_request.contents[-1]
        if last.role == "user" and last.parts and last.parts[0].text:
            logger.info(
                "[%s] USER » %s",
                callback_context.agent_name,
                last.parts[0].text.strip(),
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
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    try:
        last = llm_request.contents[-1]
        if last.role == "user" and last.parts and last.parts[0].text:
            user_text = last.parts[0].text.strip()
            text_check = check_user_input(user_text)
            if text_check:
                return text_check
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