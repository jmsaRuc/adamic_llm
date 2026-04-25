from collections.abc import AsyncGenerator

from google.cloud.translate_v3 import TranslationServiceAsyncClient
from starlette.requests import Request


async def get_google_translate_client(
    request: Request,
) -> AsyncGenerator[TranslationServiceAsyncClient]:  # pragma: no cover
    """
    Returns Google Translate client.

    You can use it like this:

    >>> async def handler(google_translate_client: TranslationServiceAsyncClient
    >>>     = Depends(get_google_translate_client)):
    >>>     result = await google_translate_client.translate_text(...)

    :param request: current request.
    :returns: Google Translate client.
    """
    return request.app.state.google_translate_client
