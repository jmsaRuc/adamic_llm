import os

from fastapi import FastAPI
from google.cloud.translate_v3 import TranslationServiceAsyncClient

from adamic_llm.settings import settings


def init_google_translate(app: FastAPI) -> None:  # pragma: no cover
    """
    Initializes Google Translate client and stores it in the application state.

    :param app: current FastAPI app.
    """
    if not settings.google_application_credentials:
        raise ValueError("Google application credentials must be provided.")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        settings.google_application_credentials
    )
    app.state.google_translate_client = TranslationServiceAsyncClient()


async def shutdown_google_translate(app: FastAPI) -> None:  # pragma: no cover
    """
    Closes Google Translate client.

    :param app: current FastAPI app.
    """
    await app.state.google_translate_client.__aexit__(None, None, None)
