"""YAML config loader with environment variable interpolation and merge logic.

Config lookup order (first directory that exists wins):
  1. Explicit ``config_dir`` argument passed to ``load_config()``
  2. ``~/.ai-craftsman-kb/``
  3. Bundled ``config/`` directory next to the repo root
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import AppConfig, FiltersConfig, SettingsConfig, SourcesConfig

logger = logging.getLogger(__name__)

# Matches ${VAR_NAME} placeholders in YAML string values.
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate_env_vars(data: Any) -> Any:
    """Recursively replace ``${VAR_NAME}`` placeholders with environment values.

    Strings containing one or more ``${...}`` tokens have those tokens
    substituted.  Missing variables produce a warning log entry and are
    replaced with an empty string so that Pydantic validation can still
    succeed (the field will be ``None`` for optional keys).

    Args:
        data: Arbitrary Python value produced by ``yaml.safe_load``.

    Returns:
        The same structure with all ``${...}`` tokens resolved.
    """
    if isinstance(data, str):

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                logger.warning("Environment variable %s is not set", var_name)
                return ""
            return value

        return _ENV_VAR_PATTERN.sub(_replace, data)

    if isinstance(data, dict):
        return {key: _interpolate_env_vars(val) for key, val in data.items()}

    if isinstance(data, list):
        return [_interpolate_env_vars(item) for item in data]

    # Scalar types (int, float, bool, None) are returned as-is.
    return data


def _find_config_dir(config_dir: Path | None) -> Path:
    """Resolve which directory contains the YAML config files.

    Config lookup order:
      1. Explicit ``config_dir`` argument (if provided and exists).
      2. ``~/.ai-craftsman-kb/`` user directory.
      3. Bundled ``config/`` directory shipped with the package (repo root).

    Args:
        config_dir: Optional explicit override.

    Returns:
        The resolved :class:`~pathlib.Path` to the config directory.
        The directory may not exist if no configuration has been created yet
        (callers should handle missing YAML files gracefully).
    """
    if config_dir is not None and config_dir.exists():
        return config_dir

    user_dir = Path.home() / ".ai-craftsman-kb"
    if user_dir.exists():
        return user_dir

    # The bundled config/ lives four levels up from this file:
    # backend/ai_craftsman_kb/config/loader.py → ../../../../config
    bundled = Path(__file__).parent.parent.parent.parent / "config"
    return bundled


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file, returning an empty dict if it does not exist.

    Uses ``yaml.safe_load`` exclusively — never ``yaml.load`` — to avoid
    arbitrary code execution from untrusted YAML.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed YAML content as a Python dict, or ``{}`` if the file is absent.
    """
    if not path.exists():
        logger.debug("Config file not found (skipping): %s", path)
        return {}

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return _interpolate_env_vars(raw)


def load_config(config_dir: Path | None = None) -> AppConfig:
    """Load and validate configuration from YAML files into an :class:`AppConfig`.

    Reads ``sources.yaml``, ``settings.yaml``, and ``filters.yaml`` from the
    resolved configuration directory, performs ``${ENV_VAR}`` interpolation,
    expands ``~`` in ``data_dir``, and validates everything through Pydantic.

    Config lookup order (first existing directory wins):
      1. ``config_dir`` argument
      2. ``~/.ai-craftsman-kb/``
      3. Bundled ``config/`` next to the repository root

    Args:
        config_dir: Optional explicit path to the directory containing the
            YAML config files.  Pass ``None`` to use the automatic lookup.

    Returns:
        A fully-validated :class:`AppConfig` instance.

    Raises:
        pydantic.ValidationError: If any YAML file contains invalid data
            that fails Pydantic schema validation.
    """
    base = _find_config_dir(config_dir)
    logger.debug("Loading config from directory: %s", base)

    sources_data = _load_yaml(base / "sources.yaml")
    settings_data = _load_yaml(base / "settings.yaml")
    filters_data = _load_yaml(base / "filters.yaml")

    # Expand ~ in data_dir at load time so the rest of the app can use it
    # as a plain string without needing to re-expand it everywhere.
    if "data_dir" in settings_data:
        settings_data["data_dir"] = str(
            Path(settings_data["data_dir"]).expanduser()
        )

    return AppConfig(
        sources=SourcesConfig(**sources_data),
        settings=SettingsConfig(**settings_data),
        filters=FiltersConfig(**filters_data),
    )


def get_provider_api_key(config: AppConfig, provider: str) -> str | None:
    """Return the API key for a named provider, or ``None`` if not configured.

    Looks up ``config.settings.providers[provider].api_key``.  Returns
    ``None`` if the provider is absent from the config or if its key is
    an empty string (which happens when the corresponding env var is unset).

    Args:
        config: The loaded application configuration.
        provider: Provider name, e.g. ``"openai"``, ``"openrouter"``.

    Returns:
        The API key string, or ``None`` if unavailable.
    """
    provider_cfg = config.settings.providers.get(provider)
    if provider_cfg is None:
        return None
    # Treat empty string (from unset env var) as absent.
    return provider_cfg.api_key or None
