"""Shared training constants for embodiment timing and scene mixing."""

EMBODIMENT_TO_IDX = {"dingo": 0, "unitree_g1": 1, "unitree_go2": 2}
EMBODIMENT_SIM_TIME = {"dingo": 0.1, "unitree_g1": 0.04, "unitree_go2": 0.04}
EMBODIMENT_CON_STEP = {"dingo": 1, "unitree_g1": 2.5, "unitree_go2": 2.5}
MAX_CON_STEP = max(EMBODIMENT_CON_STEP.values())
VIDEO_FPS_BASE = 20
