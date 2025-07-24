# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from abc import ABC, abstractmethod

from omni.isaac.core.utils.types import ArticulationAction


class BaseController(ABC):
    """[summary]

    Args:
        name (str): [description]
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationAction:
        """A controller should take inputs and returns an ArticulationAction to be then passed to the
           ArticulationController.

        Args:
            observations (dict): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            ArticulationAction: [description]
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets state of the controller."""
        return
