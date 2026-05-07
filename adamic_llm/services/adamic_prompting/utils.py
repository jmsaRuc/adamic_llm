import logging
import uuid
from typing import Any

import pycountry
from google.cloud.translate_v3 import TranslationServiceAsyncClient
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from adamic_llm.services.adamic_prompting.state import SummaryState

log = logging.getLogger("adamic_prompting.graph")

# for dev mode, to use in memory history instead of redis


async def get_config_value(value: Any) -> str:
    """
    Convert configuration values to string format, handling both string and enum types.

    Args:
        value (Any): The configuration value to process. Can be a string or an Enum.

    Returns:
        str: The string representation of the value.

    Examples:
        >>> get_config_value("tavily")
        'tavily'
        >>> get_config_value(SearchAPI.TAVILY)
        'tavily'
    """
    return value if isinstance(value, str) else value.value


async def strip_thinking_tokens(text: str) -> str:
    """
    Remove <think> and </think> tags and their content from the text.

    Iteratively removes all occurrences of content enclosed in thinking tokens.

    Args:
        text (str): The text to process

    Returns:
        str: The text with thinking tokens and their content removed
    """
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


async def format_sources(search_results: dict[str, Any]) -> str:
    """
    Format search results into a bullet-point list of sources with URLs.

    Creates a bulleted list of search results with title and URL for each source.
    Additionally, if a source has supplementary content,
    it includes these entries indented
    beneath the main source.

    Args:
        search_results (Dict[str, Any]): Search response containing a 'results' key with
                                         a list of search result objects.

    Returns:
        str: Formatted string with sources as bullet points.
            Main sources are formatted as
             "* title : url" and supplementary sources as "  * title : url".
    """
    lines = []

    # Iterate through each source in the search results
    # and format them into a bullet-point list
    for source in search_results.get("results", []):
        if source.get("title"):
            lines.append(f"* {source['title']} : {source['url']}")
            if source.get("suplementary_content"):
                for supplementary in source["suplementary_content"]:
                    # check if the document type is one of the following:
                    # 'Senere ændringer til forskriften'
                    # 'Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov'
                    if (
                        supplementary["documentType"]
                        == "Senere ændringer til forskriften"
                        or supplementary["documentType"]
                        == "Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov"
                    ):
                        lines.append(
                            f"  * {supplementary['title']} : {supplementary['url']}",
                        )
    return "\n".join(lines)


async def get_lang_name(country_code: str) -> str | None:
    """
    Get the native name of a language based on its ISO 639-1 country code.

    Args:
        country_code (str): The ISO 639-1 country code for the language.
    Returns:
        str | None:
            The native name of the language corresponding to the provided country code,
            or None if the language is not found.
    """
    language = pycountry.languages.get(alpha_2=country_code)
    if language is None:
        log.warning(
            f"Language with country code '{country_code}'"
            " not found. Returning country code as fallback."
        )
        return None

    return language.name


async def get_lang_name_from_google_cloud(
    country_code: str,
    parent: str,
    google_translate_client: TranslationServiceAsyncClient,
) -> str:
    """
    Get the native name of a language based on its ISO 639-1 country code.

    Args:
        country_code (str): The ISO 639-1 country code for the language.
        parent (str):
            The parent resource name in the format
            "projects/{project_id}/locations/global".
        google_translate_client (TranslationServiceAsyncClient):
            An instance of the Google Translate client.

    Returns:
        str: The native name of the language corresponding to the provided country code,
             or the country code itself if the language is not found or an error occurs.
    """
    try:
        response = await google_translate_client.get_supported_languages(
            display_language_code=country_code, parent=parent
        )
        for language in response.languages:
            if language.language_code == country_code:
                return language.display_name
        log.warning(
            f"Language with country code '{country_code}' not found in Google Cloud."
            " Returning country code as fallback."
        )
        return country_code
    except Exception as e:
        log.error(
            f"Error retrieving supported languages from Google Cloud: {e}"
            " Returning country code as fallback."
        )
        return country_code


async def get_latest_human_message(state: SummaryState) -> HumanMessage | None:
    """
    Retrieve the latest human message from the state.

    Returns:
        HumanMessage | None: The latest human message
            in the conversation, or None if no human messages.
    """
    message = state["messages"][-1] if state["messages"] else None

    if message is None:
        return None
    if message.type != "human":
        return None

    return message


async def get_content_from_message(message: AnyMessage) -> str | None:
    """
    Retrieve the content from a message.

    Args:
        message (AnyMessage): The message to extract content from.

    Returns:
        str | None: The content of the message, or None if no content is available.
    """
    if not message:
        return None
    if not hasattr(message, "text"):
        return None
    content = message.text if message.text else None
    if not content or not isinstance(content, str):
        return None
    return content


async def get_system_message(state: SummaryState) -> SystemMessage | None:
    """
    Retrieve the latest system message from the state.

    Returns:
        SystemMessage | None: The latest system message
            in the conversation, or None if no system messages.
    """
    if not state["messages"] or len(state["messages"]) <= 0:
        return None

    message = state["messages"][0]

    if message is None:
        return None
    if message.type != "system":
        return None

    return message


async def insert_name_into_ai_message(ai_message: AIMessage, name: str) -> AIMessage:
    """
    Insert the provided name into the AI message.

    Args:
        ai_message (AIMessage): The AI message to update.
        name (str): The name to insert into the AI message.
    Returns:
        AIMessage: The updated AI message with the name inserted.

    """
    if ai_message is None:
        raise ValueError("AI message must not be None.")
    if ai_message.type != "ai":
        raise ValueError("Provided message is not an AI message.")
    if not name:
        raise ValueError("Name must not be empty.")

    ai_message.name = name

    return ai_message


async def get_request_message_history(state: SummaryState) -> list[AnyMessage] | None:
    """Retrieve the history of human messages, excluding the latest human message.

    Returns:
        list[AnyMessage] | None:
            The history of human messages
            in the conversation, excluding the latest human message,
            or None if no human messages.
    """

    if not state["messages"] or len(state["messages"]) <= 0:
        raise ValueError(
            "State messages can not be empty when retrieving request message history."
        )
    messages = state["messages"]
    if messages[0].type != "system" and len(messages) <= 1:
        return None
    if messages[0].type == "system" and len(messages) <= 2:
        return None

    if messages[0].type == "system":
        messages_no_system = messages[1:]

        return messages_no_system[:-1] if messages_no_system else None

    return messages[:-1] if messages else None


async def get_history_key_lang_from_request_message_history(
    request_message_history: list[AnyMessage],
) -> tuple[str, str]:
    """Extract the history key and detected language code from request message history.

    Args:
    request_message_history (list[AnyMessage]):
        The history of messages in the current conversation, excluding
        the latest message and system messages.
        Expected to contain pairs of human and AI messages.

    Returns:
    tuple[str, str]:
        A tuple containing the extracted history key and detected language code.
    """
    if request_message_history is None or len(request_message_history) <= 0:
        raise ValueError(
            "Request message history is empty."
            "Cannot extract history key without any messages in the history."
        )

    if request_message_history[-1].type != "ai":
        raise ValueError(
            "Latest message in request message history is not an AI message."
            "History key extraction expects the latest message to be an AI message."
        )
    if request_message_history[-1].name is None:
        raise ValueError(
            "Latest AI message does not have a name to extract history key from."
        )

    key_lang = request_message_history[-1].name
    key, lang_code = (
        key_lang.split("_") if key_lang and "_" in key_lang else (None, None)
    )

    if not key:
        raise ValueError(
            "Could not extract key from message name"
            f" '{key_lang}'. Expected format 'key_langcode'."
        )
    if not lang_code:
        raise ValueError(
            "Could not extract language code from message name"
            f" '{key_lang}'. Expected format 'key_langcode'."
        )
    return key, lang_code


async def create_key_lang(key: str, detected_language_code: str) -> str:
    """
    Create a key for message name based on the detected language code, and key.

    Args:
        key (str): The base key for Redis message history.
        detected_language_code (str):
            The ISO 639-1 country code for the detected language.
    Returns:
        str: A unique key for message name in the format 'key_detected_language_code'.
    """
    if not key:
        raise ValueError(
            "Base key must not be empty when creating new unique key for Redis."
        )
    if not detected_language_code:
        raise ValueError(
            "Detected language code must not be empty,"
            " when creating new unique key for Redis."
        )

    return f"{key}_{detected_language_code}"


async def create_new_unique_key() -> str:
    """
    Create a new unique key for Redis without language code.

    Returns:
    str: A unique key for Redis in the format 'uuid'.
    """

    return f"{uuid.uuid4()}"
