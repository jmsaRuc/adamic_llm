import logging
from typing import Any

import pycountry
from langchain_core.messages import HumanMessage, SystemMessage

from adamic_llm.services.adamic_prompting.state import SummaryState

log = logging.getLogger("adamic_prompting.graph")


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


async def get_native_name(contry_code: str) -> str:
    """
    Get the native name of a language based on its ISO 639-1 country code.

    Args:
        contry_code (str): The ISO 639-1 country code for the language.
    Returns:
        str: The native name of the language corresponding to the provided country code.
    """

    language = pycountry.languages.get(alpha_2=contry_code)
    if language is None:
        log.warning(
            f"Language with country code '{contry_code}'"
            " not found. Returning country code as fallback."
        )
        return contry_code

    return language.name


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


async def get_latest_system_message(state: SummaryState) -> SystemMessage | None:
    """
    Retrieve the latest system message from the state.

    Returns:
        SystemMessage | None: The latest system message
            in the conversation, or None if no system messages.
    """
    message = state["messages"][-2] if len(state["messages"]) >= 2 else None

    if message is None:
        return None
    if message.type != "system":
        return None

    return message
