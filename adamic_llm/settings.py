import enum
import importlib.util
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Annotated

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    PlainValidator,
    TypeAdapter,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from yarl import URL

AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)

CustomHttpUrlStr = Annotated[
    str,
    PlainValidator(AnyHttpUrlAdapter.validate_strings),
    AfterValidator(lambda x: str(x).rstrip("/")),
]


TEMP_DIR = Path(gettempdir())


class LogLevel(enum.StrEnum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO
    # Variables for the database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "adamic_llm"
    db_pass: str = "adamic_llm"  # noqa: S105
    db_base: str = "admin"
    db_echo: bool = False

    # Variables for Redis
    redis_host: str = "adamic_llm-redis"
    redis_port: int = 6379
    redis_user: str | None = None
    redis_pass: str | None = None
    redis_base: int | None = None

    # This variable is used to define
    # multiproc_dir. It's required for [uvi|guni]corn projects.
    prometheus_dir: Path = TEMP_DIR / "prom"

    # graph varibels
    llm_provider: str = "groq"

    # Groq
    groq_llm: str = "openai/gpt-oss-120b"
    groq_api_key: str | None = None
    groq_api_base: str = "https://api.groq.com/"

    # google
    google_project_id: str | None = None
    google_application_credentials: str | None = None

    @property
    def db_url(self) -> URL:
        """
        Assemble database URL from settings.

        :return: database URL.
        """
        return URL.build(
            scheme="postgresql",
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pass,
            path=f"/{self.db_base}",
        )

    @property
    def redis_url(self) -> URL:
        """
        Assemble REDIS URL from settings.

        :return: redis URL.
        """
        path = ""
        if self.redis_base is not None:
            path = f"/{self.redis_base}"
        return URL.build(
            scheme="redis",
            host=self.redis_host,
            port=self.redis_port,
            user=self.redis_user,
            password=self.redis_pass,
            path=path,
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ADAMIC_LLM_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENABLE_LANGFUSE: bool = False

    @field_validator("ENABLE_LANGFUSE")
    @classmethod
    def check_langfuse_settings(cls, v: bool) -> bool:
        """Validate Langfuse settings if enabled."""
        if v is False:
            return v

        # Check if langfuse package is installed
        if importlib.util.find_spec("langfuse") is None:
            raise RuntimeError(
                "Langfuse is enabled but the 'langfuse' package"
                " is not installed. Please install it, e.g.,"
                " with `uv add langgraph-openai-serve[tracing]`."
            )

        # Check for required environment variables
        required_env_vars = [
            "LANGFUSE_HOST",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ]
        missing_vars = [var for var in required_env_vars if os.getenv(var) is None]

        if missing_vars:
            raise RuntimeError(
                "Langfuse is enabled but the following"
                " environment variables are not set: "
                f"{', '.join(missing_vars)}."
                " Please set these variables."
            )

        return v


settings = Settings()
