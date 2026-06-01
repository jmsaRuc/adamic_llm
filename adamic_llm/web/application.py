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
    adamic_graph_with_config_gpt_oss = adamic_graph.with_config(
        configurable={
            "llm_provider": "groq",
            "groq_llm": "openai/gpt-oss-120b",
            "groq_api_base": "https://api.groq.com/",
            "groq_api_key": settings.groq_api_key,
            "open_router_api_key": settings.openrouter_api_key,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "mode": "adamic",
        }
    )
    adamic_graph_self_translate_with_config_gpt_oss = adamic_graph.with_config(
        configurable={
            "llm_provider": "groq",
            "groq_llm": "openai/gpt-oss-120b",
            "groq_api_base": "https://api.groq.com/",
            "groq_api_key": settings.groq_api_key,
            "open_router_api_key": settings.openrouter_api_key,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "mode": "adamic",
            "translation_method": "self-translate",
        }
    )
    adamic_graph_with_config_llama4 = adamic_graph.with_config(
        configurable={
            "llm_provider": "groq",
            "groq_llm": "meta-llama/llama-4-scout-17b-16e-instruct",
            "groq_api_base": "https://api.groq.com/",
            "groq_api_key": settings.groq_api_key,
            "open_router_api_key": settings.openrouter_api_key,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "mode": "adamic",
        }
    )
    adamic_graph_self_translate_with_config_llama4 = adamic_graph.with_config(
        configurable={
            "llm_provider": "groq",
            "groq_llm": "meta-llama/llama-4-scout-17b-16e-instruct",
            "groq_api_base": "https://api.groq.com/",
            "groq_api_key": settings.groq_api_key,
            "open_router_api_key": settings.openrouter_api_key,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "mode": "adamic",
            "translation_method": "self-translate",
        }
    )
    adamic_graph_with_config_opus = adamic_graph.with_config(
        configurable={
            "llm_provider": "open-router",
            "open_router_llm": "anthropic/claude-opus-4.7",
            "open_router_api_base": "https://openrouter.ai/api/v1",
            "open_router_api_key": settings.openrouter_api_key,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "mode": "adamic",
        }
    )
    adamic_graph_self_translate_with_config_opus = adamic_graph.with_config(
        configurable={
            "llm_provider": "open-router",
            "open_router_llm": "anthropic/claude-opus-4.7",
            "open_router_api_base": "https://openrouter.ai/api/v1",
            "open_router_api_key": settings.openrouter_api_key,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "mode": "adamic",
            "translation_method": "self-translate",
        }
    )
    adamic_graph_with_config_deepseek = adamic_graph.with_config(
        configurable={
            "llm_provider": "open-router",
            "open_router_llm": "deepseek/deepseek-v4-pro",
            "open_router_api_base": "https://openrouter.ai/api/v1",
            "open_router_api_key": settings.openrouter_api_key,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "mode": "adamic",
        }
    )
    adamic_graph_self_translate_with_config_deepseek = adamic_graph.with_config(
        configurable={
            "llm_provider": "open-router",
            "open_router_llm": "deepseek/deepseek-v4-pro",
            "open_router_api_base": "https://openrouter.ai/api/v1",
            "open_router_api_key": settings.openrouter_api_key,
            "google_project_id": settings.google_project_id,
            "google_application_credentials": settings.google_application_credentials,
            "mode": "adamic",
            "translation_method": "self-translate",
        }
    )
    os.environ["GROQ_API_KEY"] = settings.groq_api_key or ""
    if not settings.openrouter_api_key:
        raise ValueError(
            "OpenRouter API key is not set. Please set it in the environment variables."
        )

    # Initialize the GraphRegistry and register the Adamic graph
    graph_registry = GraphRegistry(
        registry={
            "adamic_graph/gpt-oss-120b": GraphConfig(
                graph=adamic_graph_with_config_gpt_oss,
                streamable_node_names=["adamic_output"],
            ),
            "adamic_graph/llama-4-scout-17b-16e-instruct": GraphConfig(
                graph=adamic_graph_with_config_llama4,
                streamable_node_names=["adamic_output"],
            ),
            "adamic_graph/claude-opus-4.7": GraphConfig(
                graph=adamic_graph_with_config_opus,
                streamable_node_names=["adamic_output"],
            ),
            "adamic_graph/deepseek-v4-pro": GraphConfig(
                graph=adamic_graph_with_config_deepseek,
                streamable_node_names=["adamic_output"],
            ),
            "adamic_graph_self_translate/gpt-oss-120b": GraphConfig(
                graph=adamic_graph_self_translate_with_config_gpt_oss,
                streamable_node_names=["adamic_output"],
            ),
            "adamic_graph_self_translate/llama-4-scout-17b-16e-instruct": GraphConfig(
                graph=adamic_graph_self_translate_with_config_llama4,
                streamable_node_names=["adamic_output"],
            ),
            "adamic_graph_self_translate/claude-opus-4.7": GraphConfig(
                graph=adamic_graph_self_translate_with_config_opus,
                streamable_node_names=["adamic_output"],
            ),
            "adamic_graph_self_translate/deepseek-v4-pro": GraphConfig(
                graph=adamic_graph_self_translate_with_config_deepseek,
                streamable_node_names=["adamic_output"],
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
