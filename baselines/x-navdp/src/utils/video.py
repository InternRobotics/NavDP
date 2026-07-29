"""Video-related helpers used by training diagnostics."""

from src.training.constants import EMBODIMENT_SIM_TIME, VIDEO_FPS_BASE


def video_fps_for_embodiment(embodiment: str) -> int:
    """Scale debug-video FPS to each embodiment's simulation step time."""
    if embodiment not in EMBODIMENT_SIM_TIME:
        return VIDEO_FPS_BASE
    ref = EMBODIMENT_SIM_TIME["dingo"]
    sim_t = EMBODIMENT_SIM_TIME[embodiment]
    return int(round(VIDEO_FPS_BASE * (ref / sim_t)))
