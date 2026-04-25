from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import psycopg_pool
from fastapi import FastAPI
from prometheus_fastapi_instrumentator.instrumentation import (
    PrometheusFastApiInstrumentator,
)

from adamic_llm.services.google_translate.lifespan import (
    init_google_translate,
    shutdown_google_translate,
)
from adamic_llm.services.redis.lifespan import init_redis, shutdown_redis
from adamic_llm.settings import settings


async def _setup_db(app: FastAPI) -> None:
    """
    Creates connection pool for timescaledb.

    :param app: current FastAPI app.
    """
    app.state.db_pool = psycopg_pool.AsyncConnectionPool(
        conninfo=str(settings.db_url), open=False
    )
    await app.state.db_pool.open(wait=True)


def setup_prometheus(app: FastAPI) -> None:  # pragma: no cover
    """
    Enables prometheus integration.

    :param app: current application.
    """
    PrometheusFastApiInstrumentator(should_group_status_codes=False).instrument(
        app,
    ).expose(app, should_gzip=True, name="prometheus_metrics")


@asynccontextmanager
async def lifespan_setup(
    app: FastAPI,
) -> AsyncGenerator[None]:  # pragma: no cover
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """

    app.middleware_stack = None
    await _setup_db(app)
    init_redis(app)
    init_google_translate(app)
    setup_prometheus(app)
    app.middleware_stack = app.build_middleware_stack()

    yield
    await app.state.db_pool.close()
    await shutdown_google_translate(app)
    await shutdown_redis(app)
