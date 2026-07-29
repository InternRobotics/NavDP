"""Shared interface for robot controllers."""

from abc import ABC, abstractmethod

from isaacsim.core.utils.types import ArticulationAction


class BaseController(ABC):
    """Base class for controllers that emit Isaac articulation actions."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Return the controller identifier."""
        return self._name

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationAction:
        """Convert controller inputs into an articulation action."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset controller state when an implementation keeps history."""
