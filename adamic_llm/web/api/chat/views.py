"""Chat completion router.

This module provides the FastAPI router for the chat completion endpoint,
implementing an OpenAI-compatible interface.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from google.cloud.translate_v3 import TranslationServiceAsyncClient
from redis.asyncio import ConnectionPool

from adamic_llm.services.chat.service import ChatCompletionService
from adamic_llm.services.google_translate.dependency import get_google_translate_client
from adamic_llm.services.graph.graph_registry import GraphRegistry
from adamic_llm.services.redis.dependency import get_redis_pool
from adamic_llm.web.api.chat.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from adamic_llm.web.api.models.views import get_graph_registry_dependency

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    service: Annotated[ChatCompletionService, Depends(ChatCompletionService)],
    graph_registry: Annotated[GraphRegistry, Depends(get_graph_registry_dependency)],
    redis_pool: ConnectionPool = Depends(get_redis_pool),
    google_translate_client: TranslationServiceAsyncClient = Depends(
        get_google_translate_client
    ),
) -> StreamingResponse | ChatCompletionResponse:
    """Create a chat completion.

    This endpoint is compatible with OpenAI's chat completion API.

    Args:
        chat_request: The parsed chat completion request.
        graph_registry: The graph registry dependency.
        service: The chat completion service dependency.
        redis_pool: The Redis connection pool dependency.
        google_translate_client: The Google Translate client dependency.
    Returns:
        A chat completion response, either as a complete response or as a stream.
    """

    logger.info(
        f"Received chat completion request for model: {chat_request.model}, "
        f"stream: {chat_request.stream}"
    )

    if chat_request.stream:
        logger.info("Streaming chat completion response")
        return StreamingResponse(
            service.stream_completion(chat_request, graph_registry),
            media_type="text/event-stream",
        )

    logger.info("Generating non-streaming chat completion response")
    response = await service.generate_completion(
        chat_request, graph_registry, redis_pool, google_translate_client
    )
    logger.info("Returning non-streaming chat completion response")
    return response
