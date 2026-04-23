from collections.abc import Awaitable
from typing import cast

from redis.asyncio import ConnectionPool, Redis

from adamic_llm.web.api.redis.schema import RedisListDTO

# for dev mode, to use in memory history instead of redis as mock
in_memory_adamic_history: dict[str, list[str]] = {}


async def get_redis_list(
    key: str,
    redis_pool: ConnectionPool,
    start: int = 0,
    end: int = -1,
) -> RedisListDTO:
    """
    Get value from redis.

    :param key: redis key, to get data from.
    :param redis_pool: redis connection pool.
    :returns: information from redis.
    """
    async with Redis(connection_pool=redis_pool) as redis:
        redis_list = await cast(
            Awaitable[list[str]], redis.lrange(name=key, start=start, end=end)
        )

    return RedisListDTO(
        key=key,
        value_list=redis_list,
    )


async def rpush_redis_list(
    redis_value: RedisListDTO,
    redis_pool: ConnectionPool,
) -> int:
    """
    Append values to a list in redis.

    :param redis_value: new value data.
    :param redis_pool: redis connection pool.
    :returns: The index of the newly added element in the list.
    """
    if redis_value.value_list is not None:
        async with Redis(connection_pool=redis_pool) as redis:
            return await cast(
                Awaitable[int], redis.rpush(redis_value.key, *redis_value.value_list)
            )
    raise ValueError("Value must not be None when pushing to Redis list.")


# for dev mode, to use in memory history instead of redis
async def dev_get_in_memory_list(
    key: str,
) -> RedisListDTO:
    """
    Get value from in-memory history.

    :param key: key to get data from.
    :returns: information from in-memory history.
    """
    value_list = in_memory_adamic_history.get(key, [])
    return RedisListDTO(
        key=key,
        value_list=value_list,
    )


async def dev_rpush_in_memory_list(
    redis_value: RedisListDTO,
) -> int:
    """
    Append values to a list in in-memory history.

    :param redis_value: new value data.
    :returns: The index of the newly added element in the list.
    """
    if redis_value.value_list is not None:
        current_list = in_memory_adamic_history.get(redis_value.key, [])
        current_list.extend(redis_value.value_list)
        in_memory_adamic_history[redis_value.key] = current_list
        return len(current_list) - 1  # return index of the last added element
    raise ValueError("Value must not be None when pushing to in-memory history list.")
