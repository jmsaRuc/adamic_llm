import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


class GraphConfig(BaseModel):
    """Configuration for a graph, including the graph itself.

    Streamable node names, and optional runtime callbacks.
    """

    graph: (
        CompiledStateGraph[Any, None, Any, Any]
        | Callable[[], Awaitable[CompiledStateGraph[Any, None, Any, Any]]]
    )
    streamable_node_names: list[str]
    runtime_callbacks: list[BaseCallbackHandler] | None = None

    async def resolve_graph(
        self,
    ) -> CompiledStateGraph[Any, None, Any, Any]:
        """
        Get the graph instance.

        Handling both direct instances and async callables.
        """
        if inspect.iscoroutinefunction(self.graph):
            return await self.graph()
        return self.graph  # type: ignore[return-value]

    class Config:
        arbitrary_types_allowed = True


class GraphRegistry(BaseModel):
    """Registry for managing multiple graph configurations."""

    registry: dict[str, GraphConfig]

    def get_graph_names(self) -> list[str]:
        """Get the names of all registered graphs."""
        return list(self.registry.keys())

    def get_graph(self, name: str) -> GraphConfig:
        """Get a graph by its name.

        Args:
            name: The name of the graph to retrieve.

        Returns:
            The graph configuration associated with the given name.

        Raises:
            ValueError: If the graph name is not found in the registry.
        """
        if name not in self.registry:
            raise ValueError(f"Graph '{name}' not found in registry.")
        return self.registry[name]
