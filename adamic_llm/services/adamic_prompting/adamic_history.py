import logging

from langchain.messages import AnyMessage, HumanMessage
from redis.asyncio import ConnectionPool

from adamic_llm.services.adamic_prompting.prompts import (
    adamic_input_user_prompt,
    adamic_output_user_prompt,
    adamic_self_translate_user_prompt,
    adamic_input_with_self_translate_user_prompt,
)
from adamic_llm.services.adamic_prompting.utils import (
    get_content_from_message,
    get_history_key_lang_from_request_message_history,
    get_lang_name,
)
from adamic_llm.web.api.redis.views import dev_get_in_memory_list, get_redis_list

log = logging.getLogger("adamic_prompting.graph")

async def get_adamic_history(
    request_message_history: list[AnyMessage],
    history_key: str,
    redis_pool: ConnectionPool | None,
    mode: str,
    translation_method: str,
) -> tuple[list[AnyMessage], list[AnyMessage]] | tuple[None, None]:
    """Retrieve the Adamic input and output history via request history.

    args:
        request_message_history (list[AnyMessage]):
            The history of messages in the current conversation,
            excluding the latest message and system messages.
            Expected to contain pairs of human and AI messages.
        history_key (str):
            The key to use for retrieving the Adamic history from Redis,
            extracted from the latest human message in the request message history.
    returns:
        Tuple containing:
        - adamic_input_history (list[AnyMessage] | None):
        The reconstructed Adamic input history as a list of AnyMessage objects,
        or None if it could not be retrieved.
        - adamic_output_history (list[AnyMessage] | None):
        The reconstructed Adamic output history as a list of AnyMessage objects,
        or None if it could not be retrieved.
    """
    if translation_method == "self-translate":
        return await get_adamic_self_translate_history(
            request_message_history=request_message_history,
            history_key=history_key,
            redis_pool=redis_pool,
            mode=mode,
        )
    else:
        return await get_adamic_history_main(
            request_message_history=request_message_history,
            history_key=history_key,
            redis_pool=redis_pool,
            mode=mode,
        )

async def get_adamic_history_main(
    request_message_history: list[AnyMessage],
    history_key: str,
    redis_pool: ConnectionPool | None,
    mode: str,
) -> tuple[list[AnyMessage], list[AnyMessage]] | tuple[None, None]:
    """Retrieve the Adamic input and output history via request history.

    args:
        request_message_history (list[AnyMessage]):
            The history of messages in the current conversation,
            excluding the latest message and system messages.
            Expected to contain pairs of human and AI messages.
        history_key (str):
            The key to use for retrieving the Adamic history from Redis,
            extracted from the latest human message in the request message history.

    returns:
        Tuple containing:
        - adamic_input_history (list[AnyMessage] | None):
        The reconstructed Adamic input history as a list of AnyMessage objects,
        or None if it could not be retrieved.
        - adamic_output_history (list[AnyMessage] | None):
        The reconstructed Adamic output history as a list of AnyMessage objects,
        or None if it could not be retrieved.
    """

    if not request_message_history or len(request_message_history) <= 0:
        raise ValueError(
            "Request message history is empty."
            " Cannot retrieve Adamic history without any messages in the history."
        )

    if request_message_history[-1].type == "human":
        log.error(
            "Latest message in request message history is a human message."
            " Adamic history retrieval expects the latest message,"
            " to be an AI message corresponding"
            " to the answer for the latest human message."
        )
        return None, None

    # get adamic input history content
    if mode != "dev":
        if redis_pool is None:
            raise ValueError(f"redis_pool must not be None when agent mode is: {mode}")
        try:
            saved_list_of = await get_redis_list(key=history_key, redis_pool=redis_pool)
        except Exception as e:
            log.error(
                "Error retrieving Adamic history from Redis"
                f" for key {history_key}: {e!s}"
            )
            return None, None
    else:
        try:
            saved_list_of = await dev_get_in_memory_list(key=history_key)
        except Exception as e:
            log.error(
                "Error retrieving Adamic history from in-memory history"
                f" for key {history_key}: {e!s}"
            )
            return None, None

    list_of_content = (
        saved_list_of.value_list if saved_list_of and saved_list_of.value_list else None
    )
    if list_of_content is None or len(list_of_content) == 0:
        log.warning(f"No Adamic input history found in Redis for key: {history_key}")
        return None, None

    if len(request_message_history) > len(list_of_content):
        raise ValueError(
            f"Length of request message history ({len(request_message_history)})"
            " grater than the length of Adamic input history,"
            f" from Redis ({len(list_of_content)})."
        )

    # reconstruct adamic input history and output history as list of AnyMessage
    adamic_input_history: list[AnyMessage] = []
    adamic_output_history: list[AnyMessage] = []
    for i, message in enumerate(request_message_history):
        content = list_of_content[i]
        if message.type == "system":
            raise ValueError(
                f"request_message_history should not contain system messages,"
                f" but a system message was found at index {i}."
            )

        if message.type == "human":
            if i + 1 > len(request_message_history) - 1:
                raise ValueError(
                    "Expected pairs of human and AI messages"
                    f" in request_message_history, but human message at index {i},"
                    " does not have a corresponding AI message following it."
                )

            key, lang_code = await get_history_key_lang_from_request_message_history(
                request_message_history
            )
            if key != history_key:
                raise ValueError(
                    f"Extracted history key '{key}' from request message history,"
                    f" does not match the provided history_key '{history_key}',"
                    " for retrieving Adamic history from Redis."
                )

            lang_name = await get_lang_name(lang_code)
            if not lang_name:
                log.warning(
                    f"Could not get language name for code '{lang_code}'"
                    " using pycountry lang code as lang_name instead."
                )
                lang_name = lang_code
            
            message_content = await get_content_from_message(message)
            user_prompt_input = adamic_input_user_prompt.format(
                question_en=content,
                question_none_en=message_content,
                detected_language=lang_name,
            )
            adamic_input_history.append(HumanMessage(content=user_prompt_input))

            next_content = list_of_content[i + 1]

            user_prompt_output = adamic_output_user_prompt.format(
                question_en=content,
                question_none_en=message_content,
                detected_language=lang_name,
                answer_en=next_content,
            )
            adamic_output_history.append(HumanMessage(content=user_prompt_output))

        elif message.type == "ai":
            ai_message_input = message.model_copy()
            ai_message_input.name = None
            ai_message_input.content = content
            adamic_input_history.append(ai_message_input)

            ai_message_output = message.model_copy()
            ai_message_output.name = None
            adamic_output_history.append(ai_message_output)
        else:
            raise ValueError(
                f"Unsupported message type '{message.type}'"
                f" in request_message_history at index {i}."
                " Only 'human' and 'ai' message types are supported."
            )

    return adamic_input_history, adamic_output_history


async def get_adamic_self_translate_history(
    request_message_history: list[AnyMessage],
    history_key: str,
    redis_pool: ConnectionPool | None,
    mode: str,
) -> tuple[list[AnyMessage], list[AnyMessage]] | tuple[None, None]:
    """Retrieve the Adamic self-translate history via request history.
    args:
        request_message_history (list[AnyMessage]):
            The history of messages in the current conversation,
            excluding the latest message and system messages.
            Expected to contain pairs of human and AI messages.
        history_key (str):
            The key to use for retrieving the Adamic history from Redis,
            extracted from the latest human message in the request message history.
    returns:
        Tuple containing:
        - adamic_self_translate_history (list[AnyMessage] | None):
            The reconstructed Adamic self-translate history as a list of AnyMessage objects,
            or None if it could not be retrieved.
        - adamic_input_with_self_translate_history (list[AnyMessage] | None):
            The reconstructed Adamic input with self-translate history as a list of AnyMessage objects,
            or None if it could not be retrieved.
    """
    
    if not request_message_history or len(request_message_history) <= 0:
        raise ValueError(
            "Request message history is empty."
            " Cannot retrieve Adamic history without any messages in the history."
        )
        
    if request_message_history[-1].type == "human":
        log.error(
            "Latest message in request message history is a human message."
            " Adamic history retrieval expects the latest message,"
            " to be an AI message corresponding"
            " to the answer for the latest human message."
        )
        return None, None
    
    # get adamic self_translate history content
    if mode != "dev":
        if redis_pool is None:
            raise ValueError(f"redis_pool must not be None when agent mode is: {mode}")
        try:
            saved_list_of = await get_redis_list(key=history_key, redis_pool=redis_pool)
        except Exception as e:
            log.error(
                "Error retrieving Adamic history from Redis"
                f" for key {history_key}: {e!s}"
            )
            return None, None
    else:
        try:
            saved_list_of = await dev_get_in_memory_list(key=history_key)
        except Exception as e:
            log.error(
                "Error retrieving Adamic history from in-memory history"
                f" for key {history_key}: {e!s}"
            )
            return None, None
        
    
    list_of_content = (
        saved_list_of.value_list if saved_list_of and saved_list_of.value_list else None
    )
    
    if list_of_content is None or len(list_of_content) == 0:
        log.warning(f"No Adamic input history found in Redis for key: {history_key}")
        return None, None
    
    # reconstruct adamic self_translate history and adamic input with self_translate history as list of AnyMessage
    adamic_self_translate_history: list[AnyMessage] = []
    adamic_input_with_self_translate_history: list[AnyMessage] = []
    for i, message in enumerate(request_message_history):
        if message.type == "system":
            raise ValueError(
                f"request_message_history should not contain system messages,"
                f" but a system message was found at index {i}."
            )
        if message.type == "human":
            content = list_of_content[i]
            if i + 1 > len(request_message_history) - 1:
                raise ValueError(
                    "Expected pairs of human and AI messages"
                    f" in request_message_history, but human message at index {i},"
                    " does not have a corresponding AI message following it."
                )
        
            key, lang_code = await get_history_key_lang_from_request_message_history(
                    request_message_history
                )
            if key != history_key:
                    raise ValueError(
                        f"Extracted history key '{key}' from request message history,"
                        f" does not match the provided history_key '{history_key}',"
                        " for retrieving Adamic history from Redis."
                    )
                    
            lang_name = await get_lang_name(lang_code)
            if not lang_name:
                log.warning(
                    f"Could not get language name for code '{lang_code}'"
                    " using pycountry lang code as lang_name instead."
                )
                lang_name = lang_code
            
            message_content = await get_content_from_message(message)
            user_prompt_self_translate = adamic_self_translate_user_prompt.format(
                    question_none_en=message_content,
                    detected_language=lang_name,
                )
            adamic_self_translate_history.append(HumanMessage(content=user_prompt_self_translate))
        
            user_prompt_input_with_self_translate = adamic_input_with_self_translate_user_prompt.format(
                    question_en=content,
                    question_none_en=message_content,
                    detected_language=lang_name,
                )
            adamic_input_with_self_translate_history.append(HumanMessage(content=user_prompt_input_with_self_translate))
        
        elif message.type == "ai":
            # content of the human message corresponding to the current AI message.
            content = list_of_content[i - 1]
            ai_message_self_translate = message.model_copy()
            ai_message_self_translate.name = None
            ai_message_self_translate.content = content
            adamic_self_translate_history.append(ai_message_self_translate)
            
            ai_message_input_with_self_translate = message.model_copy()
            ai_message_input_with_self_translate.name = None
            adamic_input_with_self_translate_history.append(ai_message_input_with_self_translate)
        else:
            raise ValueError(
                f"Unsupported message type '{message.type}'"
                f" in request_message_history at index {i}."
                " Only 'human' and 'ai' message types are supported."
            )
    
    return adamic_self_translate_history, adamic_input_with_self_translate_history