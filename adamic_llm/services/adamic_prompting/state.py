from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState


@dataclass(kw_only=True)
class SummaryState(MessagesState):
    """
    State class for managing the summary of a conversation.

    Including the original question, system message,
    detected language, and answers in both English
    and non-English formats.
    """

    # Original question in none English and English (if detected)
    human_message: HumanMessage
    # human message content in none English and English (if detected)
    question_none_en: str
    question_en: str

    # System message content (if any)
    system_message: SystemMessage | None
    system_message_content: str | None

    # Detected language of the input question
    detected_language_code: str | None
    detected_language_name: str | None

    # Answer in English and none English (if detected)
    answer_en: AIMessage | None
    answer_none_en: AIMessage | None
