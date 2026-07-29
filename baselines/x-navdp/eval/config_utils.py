"""Utilities for loading evaluation YAML configs as attribute namespaces."""

from argparse import Namespace

import yaml


def _dict_to_namespace(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(item) for item in obj]
    return obj


def load_default_config(config_file: str):
    """Load an evaluation YAML file as an attribute-style namespace."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _dict_to_namespace(config)
