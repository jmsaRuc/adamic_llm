import logging
import os
from typing import Literal

from google.api_core.retry_async import AsyncRetry
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
    adamic_output_user_prompt,
)
from adamic_llm.services.adamic_prompting.state import (
    SummaryState,
)
from adamic_llm.services.adamic_prompting.utils import (
    get_latest_human_message,
    get_latest_system_message,
    get_native_name,
)

# Set up logging for the graph
log = logging.getLogger("adamic_prompting.graph")


# Nodes
async def main_router(
    state: SummaryState, config: RunnableConfig
) -> Literal["detect_language", "adamic_bypass"]:
    """Main router to determine the flow based on the detected language."""

    configurable = Configuration.from_runnable_config(config)

    mode = configurable.mode

    if mode == "direct":
        log.info("Routing to adamic bypass node since mode is set to direct")
        return "adamic_bypass"
    return "detect_language"


async def detect_language(
    state: SummaryState, config: RunnableConfig
) -> dict[str, str | HumanMessage]:
    """
    Detects the language of the input question.

    Parameters:
        state (SummaryState): The summary state containing the input question.
    Returns:
        dict[str, str | HumanMessage]:
        Updated state with the detected language.

    """
    human_message = await get_latest_human_message(state)
    if not human_message:
        raise ValueError("No human messages given")

    question_none_en = str(human_message.text) if human_message.text else None
    if not question_none_en:
        raise ValueError("The content of the human message must not be None")

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

    retry = AsyncRetry(initial=0.5, maximum=30, multiplier=2, timeout=120)
    response = await google_client.detect_language(
        parent=parent, content=question_none_en, mime_type="text/plain", retry=retry
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
        "human_message": human_message,
        "question_none_en": question_none_en,
        "detected_language_code": detected_language_code,
        "detected_language_name": detected_language_name,
    }


async def route_to_translation(
    state: SummaryState,
) -> Literal["translate_question", "adamic_bypass"]:
    """
    Routes the flow based on the detected language.

    Parameters:
        state (SummaryState): The summary state containing the detected language code.
    Returns:
        Literal["translate_question", "adamic_bypass"]: The next node to route to.
    """

    detect_language_code = state["detected_language_code"]
    detect_language_name = state["detected_language_name"]

    if detect_language_code != "en":
        log.info(
            "Routing to translation node since"
            f" detected language is {detect_language_code}"
        )
        return "translate_question"
    log.info(
        f"Routing to bypass node since detected language is {detect_language_name}"
    )
    return "adamic_bypass"


async def adamic_bypass(
    state: SummaryState,
    config: RunnableConfig,
) -> dict[str, AIMessage]:
    """
    Bypass the translation steps.

    Directly routes the original question to the output.

    Parameters:
        state (SummaryState): The summary state containing
            the original question.
    Returns:
        dict[str, MessagesState]: Updated state with the
            original question routed to the output.
    """

    message = state["messages"]

    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    llm_translate_q: ChatGroq | ChatOpenAI | ChatOllama
    if configurable.llm_provider == "groq":
        llm_translate_q = ChatGroq(
            model=configurable.groq_llm,
            temperature=0.0,
            max_tokens=8192,
            model_kwargs={"top_p": 1.0},
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

    # Run the LLM to process the original question without translation
    result = await llm_translate_q.ainvoke(message)

    return {
        "messages": result,
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

    configurable = Configuration.from_runnable_config(config)

    if (
        not configurable.google_project_id
        or not configurable.google_application_credentials
    ):
        raise ValueError(
            "Google Cloud Project ID and application credentials",
            "must be provided in the configuration",
        )

    contents = state["question_none_en"]

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        configurable.google_application_credentials
    )
    google_client = translate_v3.TranslationServiceAsyncClient()
    parent = f"projects/{configurable.google_project_id}/locations/global"
    retry = AsyncRetry(initial=0.5, maximum=30, multiplier=2, timeout=120)
    response = await google_client.translate_text(
        parent=parent,
        contents=[contents],
        mime_type="text/plain",
        source_language_code=state["detected_language_code"],
        target_language_code="en",
        retry=retry,
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
    question_en = state["question_en"]
    question_none_en = state["question_none_en"]

    detected_language = (
        state["detected_language_name"]
        if state["detected_language_name"]
        else state["detected_language_code"]
    )

    system_message = await get_latest_system_message(state)

    if system_message:
        system_message_content = (
            str(system_message.text) if system_message.text else None
        )
        log.info(f"Additional system message added: {system_message_content}")
    else:
        system_message_content = None
        log.info("No additional system message found")

    # format system prompt with the detected language
    systems_prompt = adamic_input_system_prompt.format(
        detected_language=detected_language,
        system_message_content=system_message_content if system_message_content else "",
    )

    # Insert the Danish research topic into the prompt
    user_prompt = adamic_input_user_prompt.format(
        question_en=question_en,
        question_none_en=question_none_en,
        detected_language=detected_language,
    )

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    llm_translate_q: ChatGroq | ChatOpenAI | ChatOllama
    if configurable.llm_provider == "groq":
        llm_translate_q = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=8192,
            model_kwargs={"top_p": 1.0},
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

    to_be_translated_answer_en = str(state["answer_en"].text)
    question_en = state["question_en"]
    question_none_en = state["question_none_en"]

    detected_language = (
        state["detected_language_name"]
        if state["detected_language_name"]
        else state["detected_language_code"]
    )

    # format system prompt with the detected language
    systems_prompt = adamic_output_system_prompt.format(
        detected_language=detected_language,
    )

    user_prompt = adamic_output_user_prompt.format(
        detected_language=detected_language,
        question_en=question_en,
        question_none_en=question_none_en,
        answer_en=to_be_translated_answer_en,
    )

    # Insert the detected language and the English answer into the prompt
    user_prompt = (
        f"Translate the following text from English to {detected_language}:"
        f"\n <User Input> \n {to_be_translated_answer_en} \n <User Input>\n\n"
    )

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    llm_translate_q: ChatGroq | ChatOpenAI | ChatOllama
    if configurable.llm_provider == "groq":
        llm_translate_q = ChatGroq(
            model=configurable.groq_llm,
            temperature=0.4,
            max_tokens=8192,
            model_kwargs={"top_p": 0.95},
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
builder.add_node("adamic_bypass", adamic_bypass)
builder.add_node("translate_question", translate_question)
builder.add_node("adamic_input", adamic_input)
builder.add_node("adamic_output", adamic_output)

# Add edges
builder.add_edge(START, "detect_language")
builder.add_conditional_edges("detect_language", route_to_translation)
builder.add_edge("adamic_bypass", END)
builder.add_edge("translate_question", "adamic_input")
builder.add_edge("adamic_input", "adamic_output")
builder.add_edge("adamic_output", END)
graph = builder.compile()
