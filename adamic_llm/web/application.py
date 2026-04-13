import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from adamic_llm.log import configure_logging
from adamic_llm.services.adamic_prompting.graph import graph as adamic_graph
from adamic_llm.services.graph.graph_registry import GraphConfig, GraphRegistry
from adamic_llm.settings import settings
from adamic_llm.web.api.router import api_router
from adamic_llm.web.lifespan import lifespan_setup
from adamic_llm.web.openai_server import LangchainOpenaiApiServe

APP_ROOT = Path(__file__).parent.parent


def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    configure_logging()
    app = FastAPI(
        title="adamic_llm",
        lifespan=lifespan_setup,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    adamic_graph_with_config = adamic_graph.with_config(
        configurable={
            "llm_provider": settings.llm_provider,
            "groq_llm": settings.groq_llm,
            "groq_api_base": settings.groq_api_base,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "bypass": False,
        }
    )
    adamic_graph_with_config_bypass = adamic_graph.with_config(
        configurable={
            "llm_provider": settings.llm_provider,
            "groq_llm": settings.groq_llm,
            "groq_api_base": settings.groq_api_base,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "bypass": True,
        }
    )

    os.environ["GROQ_API_KEY"] = settings.groq_api_key or ""

    # Initialize the GraphRegistry and register the Adamic graph
    graph_registry = GraphRegistry(
        registry={
            "adamic_graph": GraphConfig(
                graph=adamic_graph_with_config, streamable_node_names=["adamic_output"]
            ),
            "adamic_graph_basic": GraphConfig(
                graph=adamic_graph_with_config_bypass,
                streamable_node_names=["adamic_bypass"],
            ),
        }
    )

    # Main router for the API.
    app.include_router(router=api_router, prefix="/v1")
    # Adds static directory.
    # This directory is used to access swagger files.
    app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")

    graph_serve = LangchainOpenaiApiServe(app=app, graphs=graph_registry)
    if graph_serve.app is None:
        raise ValueError("Failed to initialize the OpenAI API server.")

    return graph_serve.app
