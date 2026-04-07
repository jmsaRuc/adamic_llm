"""LangGraph OpenAI API Serve.

This module provides a server class that connects LangGraph
instances to an OpenAI-compatible API. It allows users to
register their LangGraph instances and expose them through
a FastAPI application.

Examples:
    >>> from langgraph_openai_serve import LangchainOpenaiApiServe
    >>> from fastapi import FastAPI
    >>> from your_graphs import simple_graph_1, simple_graph_2
    >>>
    >>> app = FastAPI(title="LangGraph OpenAI API")
    >>> graph_serve = LangchainOpenaiApiServe(
    ...     app=app,
    ...     graphs={
    ...         "simple_graph_1": simple_graph_1,
    ...         "simple_graph_2": simple_graph_2
    ...     }
    ... )
    >>> graph_serve.bind_openai_chat_completion(prefix="/v1")
"""

import logging

from fastapi import FastAPI

from adamic_llm.services.graph.graph_registry import GraphRegistry

# Reorder imports
from adamic_llm.web.api.chat import views as chat_views
from adamic_llm.web.api.models import views as models_views
from adamic_llm.web.api.monitoring import views as health_views

logger = logging.getLogger(__name__)


class LangchainOpenaiApiServe:
    """Server class to connect LangGraph instances with an OpenAI-compatible API.

    This class serves as a bridge between LangGraph instances
    and an OpenAI-compatible API. It allows users to register
    their LangGraph instances and expose them through a FastAPI
    application.

    Attributes:
        app: The FastAPI application to attach routers to.
        graphs: A GraphRegistry instance containing the graphs to serve.
    """

    def __init__(
        self,
        app: FastAPI | None = None,
        graphs: GraphRegistry | None = None,
    ) -> None:
        """Initialize the server with a FastAPI app and a GraphRegistry.

        Args:
            app: The FastAPI application to attach routers to.
                If None, a new FastAPI app will be created.
            graphs: A GraphRegistry instance containing the
                graphs to serve. If None, a default simple
                graph will be used.
            configure_cors: Optional; Whether to configure
                CORS for the FastAPI application.
        """
        self.app = app

        if app is None:
            app = FastAPI(
                title="LangGraph OpenAI Compatible API",
                description="An OpenAI-compatible API for LangGraph",
                version="0.0.1",
            )
        self.app = app

        if graphs is None:
            logger.info("Graphs not provided")
            raise ValueError(
                "No graphs provided. Please provide a"
                " GraphRegistry instance with your graphs."
            )

        if isinstance(graphs, GraphRegistry):
            logger.info("Using provided GraphRegistry instance")
            self.graph_registry = graphs
        else:
            raise TypeError(
                "Invalid type for graphs parameter. Expected GraphRegistry or None."
            )

        # Attach the registry to the app's state for dependency injection
        self.app.state.graph_registry = self.graph_registry

        logger.info(
            "Initialized LangchainOpenaiApiServe with"
            f" {len(self.graph_registry.registry)} graphs"
        )
        logger.info(
            f"Available graphs: {', '.join(self.graph_registry.get_graph_names())}"
        )

    def bind_openai_chat_completion(
        self, prefix: str = "/v1"
    ) -> "LangchainOpenaiApiServe":
        """Bind OpenAI-compatible chat completion endpoints to the FastAPI app.

        Args:
            prefix: Optional; The URL prefix for the
                OpenAI-compatible endpoints. Defaults to "/v1".
        """
        if not self.app:
            raise ValueError("FastAPI app is not initialized")
        self.app.include_router(chat_views.router, prefix=prefix)
        self.app.include_router(health_views.router, prefix=prefix)
        self.app.include_router(models_views.router, prefix=prefix)

        logger.info(f"Bound OpenAI chat completion endpoints with prefix: {prefix}")

        return self
