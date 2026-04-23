from pydantic import BaseModel


class RedisListDTO(BaseModel):
    """DTO for redis list values."""

    key: str
    value_list: list[str] | None
