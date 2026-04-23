import logging
import os
from typing import Literal

from google.api_core.retry_async import AsyncRetry
from google.cloud import translate_v3
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from adamic_llm.services.adamic_prompting.adamic_history import get_adamic_history
from adamic_llm.services.adamic_prompting.configuration import Configuration
from adamic_llm.services.adamic_prompting.prompts import (
    adamic_input_system_prompt,
    adamic_input_user_prompt,
    adamic_output_system_prompt,
    adamic_output_user_prompt,
)
from adamic_llm.services.adamic_prompting.state import (
    SummaryState,
    SummaryStateInput,
)
from adamic_llm.services.adamic_prompting.utils import (
    create_key_lang,
    create_new_unique_key,
    get_content_from_message,
    get_history_key_lang_from_request_message_history,
    get_lang_name,
    get_latest_human_message,
    get_request_message_history,
    get_system_message,
    insert_name_into_ai_message,
)
from adamic_llm.web.api.redis.schema import RedisListDTO
from adamic_llm.web.api.redis.views import dev_rpush_in_memory_list, rpush_redis_list

# Set up logging for the graph
log = logging.getLogger("adamic_prompting.graph")


# Nodes
async def preprocess_input(
    state: SummaryState, config: RunnableConfig
) -> dict[str, str | HumanMessage | list[AnyMessage] | bool | None]:
    """
    Preprocesses the input question.

    Parameters:
        state (SummaryState): The summary state containing the input question.
        config (RunnableConfig): Configuration specifying the Google Cloud project ID
        and application credentials.
    Returns:
        dict[str, str | HumanMessage | list[AnyMessage] | bool]:
            Updated state with the preprocessed question.
    """
    configurable = Configuration.from_runnable_config(config)

    mode = configurable.mode

    if mode == "direct":
        log.info("Preprocessing skipped since mode is set to direct")
        return {
            "human_message": None,
            "question_none_en": None,
            "request_message_history": None,
            "message_history": False,
            "adamic_input_history": None,
            "adamic_output_history": None,
        }

    # Message history preprocessing
    adamic_input_history: list[AnyMessage] | None = None
    adamic_output_history: list[AnyMessage] | None = None

    request_message_history = await get_request_message_history(state)
    if request_message_history is None:
        message_history = False
        log.info(
            "No request message history found,"
            " proceeding without message history for Adamic graph"
        )
    else:
        message_history = True
        log.info(
            "Request message history found with"
            f" {len(request_message_history)} messages"  # type: ignore
        )

    if message_history:
        log.info(
            "Preprocessing input with message history of"
            f" {len(request_message_history)} messages"  # type: ignore
        )
        if request_message_history is None:  # for mypy
            raise ValueError(
                "request_message_history is None when message_history is True"
            )

        history_key, _ = await get_history_key_lang_from_request_message_history(
            request_message_history
        )
        log.info(
            f"Extracted history key '{history_key}'"
            " from request message history for retrieving Adamic history from Redis"
        )
        redis_pool = state.get("redis_pool", None)
        adamic_input_history, adamic_output_history = await get_adamic_history(
            request_message_history, history_key, redis_pool, mode
        )

        if adamic_input_history is None or adamic_output_history is None:
            message_history = False
            log.error(
                "No useable Adamic history retrieved from Redis,"
                " proceeding without message history for Adamic graph"
            )
        else:
            log.info(
                "Successfully retrieved Adamic input and output history, input len:"
                f" {len(adamic_input_history)} output len: {len(adamic_output_history)}"
            )

    if not message_history:
        log.info(
            "Creating new unique history key for Redis since no message history found"
        )
        history_key = await create_new_unique_key()

    human_message = await get_latest_human_message(state)
    if not human_message:
        raise ValueError("No human messages given")

    question_none_en = await get_content_from_message(human_message)
    if not question_none_en:
        raise ValueError("The content of the human message must not be None")

    return {
        "human_message": human_message,
        "question_none_en": question_none_en,
        "request_message_history": request_message_history,
        "message_history": message_history,
        "history_key": history_key,
        "adamic_input_history": adamic_input_history,
        "adamic_output_history": adamic_output_history,
    }


async def router_main(
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
) -> dict[str, str]:
    """
    Detects the language of the input question using Google Cloud Translation API.

    Parameters:
        state (SummaryState): The summary state containing the input question.
        config (RunnableConfig): Configuration specifying the Google Cloud project ID
        and application credentials.
    Returns:
        dict[str, str]: Updated state with the detected language code and name.

    """
    question_none_en = state["question_none_en"]
    if not question_none_en:
        raise ValueError("question_none_en must not be None for language detection")

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

    detected_language_name = await get_lang_name(detected_language_code)

    log.info(
        f"Detected language: Code: {detected_language_code},"
        f"Name: {detected_language_name} with confidence {detect_language_confidence}"
    )

    if detect_language_confidence < 0.5:
        log.warning(
            f"Low confidence in language detection: {detect_language_confidence}"
        )

    return {
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
    if not detect_language_code:
        raise ValueError("detected_language_code must not be None for routing decision")

    detect_language_name = state["detected_language_name"]
    if not detect_language_name:
        raise ValueError("detected_language_name must not be None for routing decision")

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
    if not message or len(message) <= 0:
        raise ValueError("No messages found in the state to bypass")

    configurable = Configuration.from_runnable_config(config)
    mode = configurable.mode

    if mode in ("adamic", "dev"):
        question_en = state["question_none_en"]
        if not question_en:
            raise ValueError(
                "question_none_en must not be None for adamic bypass node"
                " in adamic or dev mode"
            )

        key = state["history_key"]
        if not key:
            raise ValueError(
                "history_key must not be None for adamic bypass node."
                " in adamic or dev mode"
            )

        redis_values = RedisListDTO(key=key, value_list=[question_en])

        if mode != "dev":
            if state["redis_pool"] is None:
                raise ValueError(
                    "redis_pool must not be None for adamic bypass node in adamic mode"
                )
            try:
                index = await rpush_redis_list(
                    redis_value=redis_values,
                    redis_pool=state["redis_pool"],
                )
                log.info(
                    "Saved human message content to Redis list with key"
                    f" {key} at index {index}"
                )
            except Exception as e:
                log.error(f"Error pushing to Redis list: {e}")
                log.error(
                    "Failed to save human message content to Redis en-message-history"
                    f" for key {key}"
                )
        else:
            try:
                index = await dev_rpush_in_memory_list(
                    redis_value=redis_values,
                )
                log.info(
                    "Saved human message content to in-memory history with key"
                    f" {key} at index {index}"
                )
            except Exception as e:
                log.error(f"Error pushing to in-memory history: {e}")
                log.error(
                    "Failed to save human message content to in-memory"
                    " history en-message-history"
                    f" for key {key}"
                )

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

    if mode in ("adamic", "dev"):
        # get language code
        detect_language_code = state["detected_language_code"]
        if not detect_language_code:
            raise ValueError(
                "detected_language_code must not be None,"
                " for adamic bypass node in adamic or dev mode"
            )

        key = state["history_key"]
        if not key:
            raise ValueError(
                "history_key must not be None for adamic bypass node"
                " in adamic or dev mode"
            )

        key_lang = await create_key_lang(key, detect_language_code)

        # save message content to redis list with key
        content = await get_content_from_message(result)
        if not content:
            content = ""
            log.warning(
                "The content of the AI message is empty, saving empty string to Redis"
            )
        redis_response_values = RedisListDTO(key=key_lang, value_list=[content])
        if mode != "dev":
            if state["redis_pool"] is None:
                raise ValueError(
                    "redis_pool must not be None for adamic bypass node in adamic mode"
                )
            try:
                index = await rpush_redis_list(
                    redis_value=redis_response_values,
                    redis_pool=state["redis_pool"],
                )
                log.info(
                    "Saved AI message content to Redis list with key"
                    f" {key_lang} at index {index}"
                )
            except Exception as e:
                log.error(f"Error pushing to Redis list: {e}")
                log.error(
                    f"Failed to save AI message content to Redis for key {key_lang}"
                )
        else:
            try:
                index = await dev_rpush_in_memory_list(
                    redis_value=redis_response_values,
                )
                log.info(
                    "Saved AI message content to in-memory history with key"
                    f" {key_lang} at index {index}"
                )
            except Exception as e:
                log.error(f"Error pushing to in-memory history: {e}")
                log.error(
                    "Failed to save AI message content"
                    f" to in-memory history for key {key_lang}"
                )

        # insert key
        result = await insert_name_into_ai_message(result, key_lang)

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
    if not contents:
        raise ValueError("question_none_en must not be None for translation node")

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
    if not question_en:
        raise ValueError("question_en must not be None for adamic input node")

    question_none_en = state["question_none_en"]
    if not question_none_en:
        raise ValueError("question_none_en must not be None for adamic input node")

    detect_language_code = state["detected_language_code"]
    if not detect_language_code:
        raise ValueError(
            "detected_language_code must not be None for adamic input node"
        )

    detected_language = (
        state["detected_language_name"]
        if state["detected_language_name"]
        else state["detected_language_code"]
    )

    system_message = await get_system_message(state)

    if system_message:
        system_message_content = await get_content_from_message(system_message)
        if not system_message_content:
            log.warning(
                "System message found but content is empty,"
                " proceeding without system message content"
            )
            system_message_content = None
        else:
            log.info(f"Additional system message added: {system_message_content}")
    else:
        system_message_content = None
        log.info("No additional system message found")

    # get history
    message_history = state["message_history"]
    adamic_input_history: list[AnyMessage] | None

    if message_history:
        adamic_input_history = state["adamic_input_history"]
        if not adamic_input_history:
            raise ValueError(
                "Message_history is True but adamic_input_history is None,"
                " in the state for adamic input node"
            )
        log.info(
            f"Using message history with {len(adamic_input_history)}"
            " messages for adamic input node"
        )
    else:
        adamic_input_history = None

    key = state["history_key"]
    if not key:
        raise ValueError("history_key must not be None for adamic input node")

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

    # save the English question to Redis list for history
    # if dev mode, save to in-memory history instead of Redis
    configurable = Configuration.from_runnable_config(config)
    mode = configurable.mode
    redis_values = RedisListDTO(key=key, value_list=[question_en])

    if mode != "dev":
        if state["redis_pool"] is None:
            raise ValueError(
                "redis_pool must not be None for adamic bypass node in adamic mode"
            )
        try:
            index = await rpush_redis_list(
                redis_values,
                redis_pool=state["redis_pool"],
            )
            log.info(
                f"Saved input user prompt to Redis list with key {key} at index {index}"
            )
        except Exception as e:
            log.error(f"Error pushing to Redis list: {e}")
            log.error(f"Failed to save input user prompt to Redis for key {key}")
    else:
        try:
            index = await dev_rpush_in_memory_list(
                redis_values,
            )
            log.info(
                "Saved input user prompt to in-memory history"
                f" with key {key} at index {index}"
            )
        except Exception as e:
            log.error(f"Error pushing to in-memory history: {e}")
            log.error(
                f"Failed to save input user prompt to in-memory history for key {key}"
            )

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

    # insert formattet new prompt and system prompt into the history
    # if message history is enabled
    input_messages: list[AnyMessage] = []

    if message_history:
        if not adamic_input_history:
            raise ValueError(
                "Message_history is True but adamic_input_history is None,"
                " in the state for adamic input node"
            )

        input_messages = adamic_input_history

    input_messages.insert(0, SystemMessage(content=systems_prompt))
    input_messages.append(HumanMessage(content=user_prompt))

    # Run the LLM
    result = await llm_translate_q.ainvoke(
        input_messages,
    )

    content = await get_content_from_message(result)
    if not content:
        content = ""
        log.warning(
            "The content of the AI message is empty, saving empty string to Redis"
        )

    # save the AI message content to Redis list for history with key
    # if dev mode, save to in-memory history instead of Redis
    redis_response_values = RedisListDTO(key=key, value_list=[content])
    if mode != "dev":
        if state["redis_pool"] is None:
            raise ValueError(
                "redis_pool must not be None for adamic bypass node in adamic mode"
            )
        try:
            index = await rpush_redis_list(
                redis_response_values,
                redis_pool=state["redis_pool"],
            )
            log.info(
                "Saved AI message content from input node to Redis list"
                f" with key {key} at index {index}"
            )
        except Exception as e:
            log.error(f"Error pushing to Redis list: {e}")
            log.error(
                "Failed to save AI message content from input node to Redis"
                f" for key {key}"
            )
    else:
        try:
            index = await dev_rpush_in_memory_list(
                redis_response_values,
            )
            log.info(
                "Saved AI message content from input node to in-memory history"
                f" with key {key} at index {index}"
            )
        except Exception as e:
            log.error(f"Error pushing to in-memory history: {e}")
            log.error(
                "Failed to save AI message content from input node to in-memory history"
                f" for key {key}"
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

    to_be_translated_answer_en = await get_content_from_message(state["answer_en"])
    if not to_be_translated_answer_en:
        raise ValueError(
            "The content of state.answer_en must not be None for adamic output node"
        )

    question_en = state["question_en"]
    if not question_en:
        raise ValueError("question_en must not be None for adamic output node")

    question_none_en = state["question_none_en"]
    if not question_none_en:
        raise ValueError("question_none_en must not be None for adamic output node")

    detected_language = (
        state["detected_language_name"]
        if state["detected_language_name"]
        else state["detected_language_code"]
    )

    # get history
    message_history = state["message_history"]
    adamic_output_history: list[AnyMessage] | None

    if message_history:
        adamic_output_history = state["adamic_output_history"]
        if not adamic_output_history:
            raise ValueError(
                "Message_history is True but adamic_output_history is None,"
                " in the state for adamic output node"
            )
        log.info(
            f"Using message history with {len(adamic_output_history)}"
            " messages for adamic output node"
        )
    else:
        adamic_output_history = None

    key = state["history_key"]
    if not key:
        raise ValueError("history_key must not be None for adamic output node")

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

    # insert formattet new prompt and system prompt into the history
    # if message history is enabled
    output_messages: list[AnyMessage] = []

    if message_history:
        if not adamic_output_history:
            raise ValueError(
                "Message_history is True but adamic_output_history is None,"
                " in the state for adamic output node"
            )
        output_messages = adamic_output_history

    output_messages.insert(0, SystemMessage(content=systems_prompt))
    output_messages.append(HumanMessage(content=user_prompt))

    # Run the LLM
    result = await llm_translate_q.ainvoke(
        output_messages,
    )

    # save message content to redis list with key
    content = await get_content_from_message(result)
    if not content:
        content = ""
        log.warning(
            "The content of the AI message is empty, saving empty string to Redis"
        )

    # Update the content of the existing AI message in the state,
    # with the new translated content.
    old_ai_answer_en = state["answer_en"]
    if not old_ai_answer_en:
        raise ValueError("state.answer_en must not be None for adamic output node")

    old_ai_answer_en.content = content

    # insert name with key into the AI message
    lang_code = state["detected_language_code"]
    if not lang_code:
        raise ValueError(
            "detected_language_code must not be None for adamic output node"
            " when inserting key into AI message"
        )

    key_lang = await create_key_lang(key, lang_code)
    old_ai_answer_en = await insert_name_into_ai_message(old_ai_answer_en, key_lang)

    return {
        "messages": old_ai_answer_en,
    }


# Add nodes and edges
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=MessagesState,
    config_schema=Configuration,
)  # type: ignore[call-arg]
builder.add_node("preprocess_input", preprocess_input)
builder.add_node("detect_language", detect_language)
builder.add_node("adamic_bypass", adamic_bypass)
builder.add_node("translate_question", translate_question)
builder.add_node("adamic_input", adamic_input)
builder.add_node("adamic_output", adamic_output)

# Add edges
builder.add_edge(START, "preprocess_input")
builder.add_conditional_edges("preprocess_input", router_main)
builder.add_conditional_edges("detect_language", route_to_translation)
builder.add_edge("adamic_bypass", END)
builder.add_edge("translate_question", "adamic_input")
builder.add_edge("adamic_input", "adamic_output")
builder.add_edge("adamic_output", END)
graph = builder.compile()
