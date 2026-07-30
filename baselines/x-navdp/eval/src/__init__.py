"""
NavDP Evaluation Module.

This module provides the server and client components for running
navigation policy evaluation with the NavDP diffusion-based planner.
"""

from .client_utils import (
    navigator_reset,
    navigator_shutdown,
    pointgoal_step,
)

__all__ = [
    "navigator_reset",
    "navigator_shutdown",
    "pointgoal_step",
]
