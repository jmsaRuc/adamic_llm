from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from redis.asyncio import ConnectionPool

from adamic_llm.services.adamic_prompting.adamic_history import get_adamic_history
from adamic_llm.services.adamic_prompting.prompts import (
    adamic_input_user_prompt,
    adamic_output_user_prompt,
)
from adamic_llm.services.adamic_prompting.state import SummaryState
from adamic_llm.services.adamic_prompting.utils import (
    create_key_lang,
    create_new_unique_key,
    get_history_key_lang_from_request_message_history,
    get_lang_name,
    get_request_message_history,
    insert_name_into_ai_message,
)
from adamic_llm.web.api.redis.schema import RedisListDTO
from adamic_llm.web.api.redis.views import rpush_redis_list


async def test_request_no_history() -> None:
    """Tests that if there is only one human message.

    The request message history should be None, as there is no history to provide.
    """
    messages: list[AnyMessage]
    messages = [HumanMessage(content="hvad er hovedstaden i midt land?")]
    state = SummaryState(messages=messages)
    request_message_history = await get_request_message_history(state)
    assert request_message_history is None


async def test_request_no_history_system_message() -> None:
    """Tests that if there is only one human message with a system message.

    The request message history should be None, as there is no history to provide.
    """
    messages: list[AnyMessage]
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="hvad er hovedstaden i midt land?"),
    ]
    state = SummaryState(messages=messages)
    request_message_history = await get_request_message_history(state)
    assert request_message_history is None


async def test_request_history() -> None:
    """Tests that if there are multiple human messages.

    The request message history is correct.
    """
    messages: list[AnyMessage]
    messages = [
        HumanMessage(content="hvad er hovedstaden i midt land?"),
        AIMessage(content="Hovedstaden i dit land er København."),
        HumanMessage(content="hvor mange indbyggere er der?"),
    ]
    state = SummaryState(messages=messages)
    request_message_history = await get_request_message_history(state)
    assert request_message_history is not None
    assert len(request_message_history) == 2
    assert isinstance(request_message_history[0], HumanMessage)
    assert request_message_history[0].content == "hvad er hovedstaden i midt land?"


async def test_request_history_with_system_message() -> None:
    """Tests that if there are multiple human messages with a system message.

    The request message history is correct.
    """
    messages: list[AnyMessage]
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="hvad er hovedstaden i midt land?"),
        AIMessage(content="Hovedstaden i dit land er København."),
        HumanMessage(content="hvor mange indbyggere er der?"),
    ]
    state = SummaryState(messages=messages)
    request_message_history = await get_request_message_history(state)
    assert request_message_history is not None
    assert len(request_message_history) == 2
    assert isinstance(request_message_history[0], HumanMessage)
    assert request_message_history[0].content == "hvad er hovedstaden i midt land?"


async def test_adamic_history(fake_redis_pool: ConnectionPool) -> None:
    """Tests that if there are multiple human messages with a system message.

    The request message history is correct, and the Adamic history is correctly
    retrieved from Redis and reconstructed into the correct format for the Adamic input
    and output nodes.
    """

    # First run, simulating the first turn in the conversation
    messages: list[AnyMessage]
    messages = [HumanMessage(content="hvad er hovedstaden i midt land?")]
    state = SummaryState(messages=messages)

    request_message_history = await get_request_message_history(state)
    assert request_message_history is None

    history_key = await create_new_unique_key()
    assert history_key is not None

    test_translated_human_content = "what is the capital of my country?"
    index = await rpush_redis_list(
        RedisListDTO(
            key=history_key,
            value_list=[test_translated_human_content],
        ),
        fake_redis_pool,
    )
    assert index == 1

    test_adamic_input_ai_content = "The capital of your country is Copenhagen."
    index2 = await rpush_redis_list(
        RedisListDTO(
            key=history_key,
            value_list=[test_adamic_input_ai_content],
        ),
        fake_redis_pool,
    )
    assert index2 == 2

    test_adamic_output_ai_content = "Hovedstaden i dit land er København."
    test_adamic_output_ai = AIMessage(content=test_adamic_output_ai_content)

    key_lang = await create_key_lang(history_key, "da")
    test_adamic_output_ai_with_name = await insert_name_into_ai_message(
        test_adamic_output_ai, key_lang
    )
    assert test_adamic_output_ai_with_name.name == key_lang

    # Second run, simulating the next turn in the conversation
    messages2: list[AnyMessage]
    messages2 = [
        HumanMessage(content="hvad er hovedstaden i midt land?"),
        test_adamic_output_ai_with_name,
        HumanMessage(content="hvor mange indbyggere er der?"),
    ]
    setate2 = SummaryState(messages=messages2)

    request_message_history2 = await get_request_message_history(setate2)

    assert request_message_history2 is not None
    assert len(request_message_history2) == 2
    assert isinstance(request_message_history2[0], HumanMessage)
    assert request_message_history2[0].content == "hvad er hovedstaden i midt land?"
    assert isinstance(request_message_history2[1], AIMessage)
    assert request_message_history2[1].content == test_adamic_output_ai_content
    assert request_message_history2[1].name == key_lang

    history_key2, lang_code = await get_history_key_lang_from_request_message_history(
        request_message_history2
    )
    assert history_key == history_key2
    adamic_input_history, adamic_output_history = await get_adamic_history(
        request_message_history2, history_key2, fake_redis_pool, mode="adamic"
    )

    lang_name = await get_lang_name(lang_code)
    assert lang_name == "Danish"

    test_adamic_input_human_content = adamic_input_user_prompt.format(
        question_en=test_translated_human_content,
        question_none_en=messages2[0].content,
        detected_language=lang_name,
    )

    test_adamic_output_human_content = adamic_output_user_prompt.format(
        question_en=test_translated_human_content,
        question_none_en=messages2[0].content,
        answer_en=test_adamic_input_ai_content,
        detected_language=lang_name,
    )

    assert adamic_input_history is not None
    assert adamic_output_history is not None
    assert len(adamic_input_history) == 2
    assert len(adamic_output_history) == 2
    assert adamic_input_history[0].content == test_adamic_input_human_content
    assert adamic_input_history[0].type == "human"
    assert adamic_input_history[1].content == test_adamic_input_ai_content
    assert adamic_input_history[1].type == "ai"
    assert adamic_output_history[0].content == test_adamic_output_human_content
    assert adamic_output_history[0].type == "human"
    assert adamic_output_history[1].type == "ai"
    assert adamic_output_history[1].content == test_adamic_output_ai_content
