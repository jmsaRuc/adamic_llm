import logging
import os
from typing import Any

from google.cloud import translate_v3
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from adamic_llm.services.adamic_prompting.configuration import Configuration
from adamic_llm.services.adamic_prompting.prompts import (
    adamic_input_system_prompt,
    adamic_input_user_prompt,
    adamic_output_system_prompt,
)
from adamic_llm.services.adamic_prompting.state import (
    SummaryState,
)
from adamic_llm.services.adamic_prompting.utils import (
    get_native_name,
)

# Set up logging for the graph
log = logging.getLogger("adamic_prompting.graph")


# Nodes
async def detect_language(
    state: SummaryState, config: RunnableConfig
) -> dict[str, str | list[str | dict[Any, Any]]]:
    """
    Detects the language of the input question.

    Parameters:
        state (SummaryState): The summary state containing the input question.
    Returns:
        dict[str, str | list[str | Dict[Any, Any]]]:
        Updated state with the detected language.

    """
    message = state["messages"][-1].content if state["messages"] else None

    if not message:
        raise ValueError("state.question_none_en must not be None")

    if (
        not isinstance(message, str)
        and message[0]
        and isinstance(message[0], dict)
        and "text" in message[0]
    ):
        message_none_en = message[0]["text"]
    elif isinstance(message, str):
        message_none_en = message
    else:
        raise ValueError(
            "state.question_none_en must be a string",
            "or a list of dictionaries with a 'text' key",
        )

    configurable = Configuration.from_runnable_config(config)

    if (
        not configurable.google_project_id
        or not configurable.google_application_credentials
    ):
        raise ValueError(
            "Google Cloud Project ID and application credentials",
            "must be provided in the configuration",
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        configurable.google_application_credentials
    )
    google_client = translate_v3.TranslationServiceAsyncClient()
    parent = f"projects/{configurable.google_project_id}/locations/global"

    # tjeck if correct credentials and project id are provided

    response = await google_client.detect_language(
        parent=parent,
        content=str(message_none_en),
        mime_type="text/plain",
    )
    detected_language_code = response.languages[0].language_code
    detect_language_confidence = response.languages[0].confidence

    detected_language_name = await get_native_name(detected_language_code)

    log.info(
        f"Detected language: Code: {detected_language_code},"
        f"Name: {detected_language_name} with confidence {detect_language_confidence}"
    )

    if detect_language_confidence < 0.5:
        log.warning(
            f"Low confidence in language detection: {detect_language_confidence}"
        )

    return {
        "question_none_en": message_none_en,
        "detected_language_code": detected_language_code,
        "detected_language_name": detected_language_name,
    }


async def translate_question(
    state: SummaryState,
    config: RunnableConfig,
) -> dict[str, str]:
    """
    Translates the input question from Danish to English using a configured LLM.

    Parameters:
        state (SummaryState): The summary state containing the Danish question.
        config (RunnableConfig): Configuration specifying the LLM provider
        and related parameters.
    Returns:
        dict[str, str]: Updated state with both the original Danish question
        and its English translation.
    """

    message_none_en = state["messages"][-1].content if state["messages"] else None
    if not message_none_en:
        raise ValueError("state.question_none_en must not be None")

    configurable = Configuration.from_runnable_config(config)

    if (
        not configurable.google_project_id
        or not configurable.google_application_credentials
    ):
        raise ValueError(
            "Google Cloud Project ID and application credentials",
            "must be provided in the configuration",
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        configurable.google_application_credentials
    )
    google_client = translate_v3.TranslationServiceAsyncClient()
    parent = f"projects/{configurable.google_project_id}/locations/global"

    response = await google_client.translate_text(
        parent=parent,
        contents=[str(message_none_en)],
        mime_type="text/plain",
        source_language_code=state["detected_language_code"],
        target_language_code="en",
    )
    translated_question = response.translations[0].translated_text

    return {
        "question_en": translated_question,
    }


async def adamic_input(
    state: SummaryState,
    config: RunnableConfig,
) -> dict[str, AIMessage]:
    """
    Asynchronously translates a Danish question to English using a configured LLM.

    Parameters:
        state (SummaryState): The summary state containing the Danish question.
        config (RunnableConfig): Configuration specifying the LLM provider
        and related parameters.
    Returns:
        dict[str, AIMessage]: Updated state with both the original Danish question
        and its English translation.
    """
    # assert if state.question_none_en is not None
    if not state["question_none_en"]:
        raise ValueError("state.question_none_en must not be None")

    # format system prompt with the detected language
    systems_prompt = adamic_input_system_prompt.format(
        detected_language=state["detected_language_name"],
    )

    # Insert the Danish research topic into the prompt
    user_prompt = adamic_input_user_prompt.format(
        question_en=state["question_en"],
        question_none_en=state["question_none_en"],
        detected_language=state["detected_language_name"],
    )

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    llm_translate_q: ChatGroq | ChatOpenAI | ChatOllama
    if configurable.llm_provider == "groq":
        llm_translate_q = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=2048,
            timeout=120,
            max_retries=5,
        )
    elif configurable.llm_provider == "openai":
        llm_translate_q = ChatOpenAI(  # type: ignore[call-arg]
            model=configurable.openai_llm,
            base_url=configurable.openai_api_base,
            temperature=0.7,
            max_tokens=2048,
            timeout=120,
            max_retries=5,
        )
    else:  # Default to Ollama
        llm_translate_q = ChatOllama(
            model=configurable.local_llm,
            temperature=0.7,
            num_predict=2048,
        )

    # Run the LLM to translate the question
    result = await llm_translate_q.ainvoke(
        [
            SystemMessage(content=systems_prompt),
            HumanMessage(content=user_prompt),
        ],
    )

    return {
        "answer_en": result,
    }


async def adamic_output(
    state: SummaryState,
    config: RunnableConfig,
) -> dict[str, AIMessage]:
    """
    Asynchronously translates a Danish question to English using a configured LLM.

    Parameters:
        state (SummaryState): The summary state containing the Danish question.
        config (RunnableConfig): Configuration specifying the LLM provider
        and related parameters.
    Returns:
        dict[str, AIMessage]: Updated state with both the original Danish question
        and its English translation.
    """
    # assert if state.answer_en is not None
    if not state["answer_en"]:
        raise ValueError("state.answer_en must not be None")

    # format system prompt with the detected language
    systems_prompt = adamic_output_system_prompt.format(
        detected_language=state["detected_language_name"],
        question_en=state["question_en"],
        question_none_en=state["question_none_en"],
    )

    to_be_translated_answer_en = str(state["answer_en"].content)

    # Insert the detected language and the English answer into the prompt
    user_prompt = (
        f"Translate the following text from English to Danish. "
        f"\n <User Input> \n {to_be_translated_answer_en} \n <User Input>\n\n"
    )

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    llm_translate_q: ChatGroq | ChatOpenAI | ChatOllama
    if configurable.llm_provider == "groq":
        llm_translate_q = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=2048,
            timeout=120,
            max_retries=5,
        )
    elif configurable.llm_provider == "openai":
        llm_translate_q = ChatOpenAI(  # type: ignore[call-arg]
            model=configurable.openai_llm,
            base_url=configurable.openai_api_base,
            temperature=0.5,
            max_tokens=2048,
            timeout=120,
            max_retries=5,
        )
    else:  # Default to Ollama
        llm_translate_q = ChatOllama(
            model=configurable.local_llm,
            temperature=0.5,
            num_predict=2048,
        )

    # Run the LLM to translate the question
    result = await llm_translate_q.ainvoke(
        [
            SystemMessage(content=systems_prompt),
            HumanMessage(content=user_prompt),
        ],
    )

    old_ai_answer_en = state["answer_en"]

    old_ai_answer_en.content = result.content

    return {
        "messages": old_ai_answer_en,
    }


# Add nodes and edges
builder = StateGraph(
    SummaryState,
    input=MessagesState,
    output=MessagesState,
    config_schema=Configuration,
)  # type: ignore[call-arg]
builder.add_node("detect_language", detect_language)
builder.add_node("translate_question", translate_question)
builder.add_node("adamic_input", adamic_input)
builder.add_node("adamic_output", adamic_output)

# Add edges
builder.add_edge(START, "detect_language")
builder.add_edge("detect_language", "translate_question")
builder.add_edge("translate_question", "adamic_input")
builder.add_edge("adamic_input", "adamic_output")
builder.add_edge("adamic_output", END)
graph = builder.compile()
