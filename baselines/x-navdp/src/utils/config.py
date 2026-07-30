"""Configuration loading helpers."""

from argparse import Namespace

import yaml


def dict_to_namespace(obj):
    """Recursively expose YAML config dictionaries as attribute namespaces."""
    if isinstance(obj, dict):
        return Namespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(item) for item in obj]
    return obj


def load_config(config_file):
    """Load a YAML config file as nested argparse namespaces."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return dict_to_namespace(config)
