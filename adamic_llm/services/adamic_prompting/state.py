from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from redis.asyncio import ConnectionPool


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class SummaryState(MessagesState, total=False):
    """
    State class for managing the summary of a conversation.

    Including the original question, system message,
    detected language, and answers in both English
    and non-English formats.
    """

    # redis
    redis_pool: ConnectionPool | None

    # Original question in none English and English (if detected)
    human_message: HumanMessage | None

    # human message content in none English and English (if detected)
    question_none_en: str | None
    question_en: str | None

    # Request message history, not including the latest message, or system.
    request_message_history: list[AnyMessage] | None
    message_history: bool | None

    # used as key for retrieving Adamic input and output history from Redis
    history_key: str | None

    # translated mirror message history for the adamic input node
    # and output node
    adamic_input_history: list[AnyMessage] | None
    adamic_output_history: list[AnyMessage] | None

    # System message content (if any)
    system_message: SystemMessage | None
    system_message_content: str | None

    # Detected language of the input question
    detected_language_code: str | None
    detected_language_name: str | None

    # Answer in English and none English (if detected)
    answer_en: AIMessage | None
    answer_none_en: AIMessage | None


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class SummaryStateInput(MessagesState, total=False):
    """Input State Class."""

    # redis pool
    redis_pool: ConnectionPool | None
