from typing import NotRequired

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from redis.asyncio import ConnectionPool


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class SummaryState(MessagesState):
    """
    State class for managing the summary of a conversation.

    Including the original question, system message,
    detected language, and answers in both English
    and non-English formats.
    """

    # redis
    redis_pool: NotRequired[ConnectionPool]

    # Original question in none English and English (if detected)
    human_message: NotRequired[HumanMessage]

    # human message content in none English and English (if detected)
    question_none_en: NotRequired[str]
    question_en: NotRequired[str]

    # Request message history, not including the latest message, or system.
    request_message_history: NotRequired[list[AnyMessage]]
    message_history: NotRequired[bool]

    # used as key for retrieving Adamic input and output history from Redis
    history_key: NotRequired[str]

    # translated mirror message history for the adamic input node
    # and output node
    adamic_input_history: NotRequired[list[AnyMessage]]
    adamic_output_history: NotRequired[list[AnyMessage]]

    # System message content (if any)
    system_message: NotRequired[SystemMessage]
    system_message_content: NotRequired[str]

    # Detected language of the input question
    detected_language_code: NotRequired[str]
    detected_language_name: NotRequired[str]

    # Answer in English and none English (if detected)
    answer_en: NotRequired[AIMessage]
    answer_none_en: NotRequired[AIMessage]


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class SummaryStateInput(MessagesState):
    """Input State Class."""

    # redis pool
    redis_pool: NotRequired[ConnectionPool]
