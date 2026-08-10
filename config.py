import os
from pathlib import Path


def _load_yaml_config():
    """Load the full config.yaml as a dict."""
    try:
        import yaml
        config_path = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_router_config():
    """Load smart_model_routing section from config.yaml."""
    try:
        data = _load_yaml_config()
        return data.get("smart_model_routing", {})
    except Exception:
        return {}


def load_default_model():
    """Read the Hermes default model from model.default in config.yaml."""
    try:
        data = _load_yaml_config()
        model_section = data.get("model", {})
        return model_section.get("default", "")
    except Exception:
        return ""
