"""Observation adapters between Isaac Lab envs and X-NavDP trainers."""


def parse_x_navdp_observations(observation):
    """Convert raw Isaac observation groups into the X-NavDP trainer input dict."""
    return {
        "rgb": observation["obs_rgb"],
        "depth": observation["obs_depth"],
        "pointgoal": observation["goal_pose"],
    }
