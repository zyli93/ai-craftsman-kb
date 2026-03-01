"""Config package — YAML loading, Pydantic validation, and env var interpolation.

Public API::

    from ai_craftsman_kb.config import load_config, AppConfig, get_provider_api_key

    config = load_config()                              # uses bundled defaults
    config = load_config(config_dir=Path("/my/config")) # explicit override
    key = get_provider_api_key(config, "openai")
"""

from .loader import get_provider_api_key, load_config
from .models import AppConfig

__all__ = ["load_config", "AppConfig", "get_provider_api_key"]
