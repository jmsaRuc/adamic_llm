from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState


@dataclass(kw_only=True)
class SummaryState(MessagesState):
    """
    A container for managing and tracking the state of a research summary.

    Including queries, research topics, results.
    from both Danish and English sources, and other related information.
    """

    # Report topic in Danish and English
    question_none_en: str | None
    question_en: str | None

    # Detected language of the input question
    detected_language_code: str | None
    detected_language_name: str | None

    # question answered in English and Danish

    answer_en: AIMessage | None
    answer_none_en: AIMessage | None
