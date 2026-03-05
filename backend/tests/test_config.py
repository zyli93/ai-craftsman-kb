"""Unit tests for the config loading system (task_02).

Covers:
- Valid config loads successfully from bundled config/
- Missing optional fields fall back to Pydantic defaults
- Environment variable interpolation via ${VAR_NAME} syntax
- Missing env vars produce a warning and an empty/None value
- Invalid field type raises pydantic.ValidationError
- get_provider_api_key returns the correct key or None
- Config lookup order: explicit dir > ~/.ai-craftsman-kb/ > bundled
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from ai_craftsman_kb.config import AppConfig, get_provider_api_key, load_config
from ai_craftsman_kb.config.loader import _interpolate_env_vars, _load_yaml
from ai_craftsman_kb.config.models import (
    FiltersConfig,
    LLMTaskConfig,
    LLMRoutingConfig,
    SettingsConfig,
    SourcesConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_settings() -> dict:
    """Return the smallest valid settings dict (only required fields)."""
    return {
        "llm": {
            "filtering": {"provider": "openrouter", "model": "llama-3-8b"},
            "entity_extraction": {"provider": "openrouter", "model": "llama-3-8b"},
            "briefing": {"provider": "anthropic", "model": "claude-3-haiku"},
            "source_discovery": {"provider": "openrouter", "model": "llama-3-8b"},
            "keyword_extraction": {"provider": "openrouter", "model": "llama-3-8b"},
        }
    }


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test: bundled config loads successfully
# ---------------------------------------------------------------------------

class TestLoadBundledConfig:
    """load_config() with no arguments should use the bundled config/ directory."""

    def test_returns_app_config_instance(self) -> None:
        """load_config() returns an AppConfig when the bundled files are valid."""
        config = load_config()
        assert isinstance(config, AppConfig)

    def test_sources_config_populated(self) -> None:
        """Bundled sources.yaml produces non-empty source lists."""
        config = load_config()
        sources = config.sources
        assert len(sources.substack) > 0
        assert len(sources.youtube_channels) > 0
        assert len(sources.subreddits) > 0
        assert len(sources.rss) > 0
        assert sources.hackernews is not None
        assert sources.arxiv is not None
        assert sources.devto is not None

    def test_settings_llm_routing_present(self) -> None:
        """Bundled settings.yaml includes all five LLM task routes."""
        config = load_config()
        llm = config.settings.llm
        assert llm.filtering.provider == "openrouter"
        assert llm.briefing.provider == "anthropic"
        assert llm.entity_extraction.provider == "openrouter"
        assert llm.source_discovery.provider == "openrouter"
        assert llm.keyword_extraction.provider == "openrouter"

    def test_embedding_defaults(self) -> None:
        """Bundled settings.yaml configures the embedding block correctly."""
        config = load_config()
        emb = config.settings.embedding
        assert emb.provider == "llamacpp"
        assert emb.model == "v5-small-retrieval-Q8_0.gguf"
        assert emb.chunk_size == 2000

    def test_server_ports(self) -> None:
        """Server ports match the expected defaults."""
        config = load_config()
        assert config.settings.server.backend_port == 8000
        assert config.settings.server.dashboard_port == 3000

    def test_filters_hackernews_enabled(self) -> None:
        """Bundled filters.yaml has HackerNews filtering enabled with LLM strategy."""
        config = load_config()
        hn = config.filters.hackernews
        assert hn.enabled is True
        assert hn.strategy == "llm"
        assert hn.min_score == 5

    def test_filters_substack_disabled(self) -> None:
        """Bundled filters.yaml disables Substack filtering by default."""
        config = load_config()
        assert config.filters.substack.enabled is False

    def test_data_dir_expanded(self) -> None:
        """data_dir has ~ expanded to the actual home directory."""
        config = load_config()
        data_dir = config.settings.data_dir
        assert not data_dir.startswith("~"), "~ should have been expanded"
        assert str(Path.home()) in data_dir


# ---------------------------------------------------------------------------
# Test: missing optional fields use Pydantic defaults
# ---------------------------------------------------------------------------

class TestMissingOptionalFields:
    """Fields absent from YAML fall back to Pydantic model defaults."""

    def test_empty_sources_uses_defaults(self, tmp_path: Path) -> None:
        """Empty sources.yaml produces empty lists and None for optional blocks."""
        _write_yaml(tmp_path / "sources.yaml", {})
        _write_yaml(tmp_path / "settings.yaml", _minimal_settings())
        _write_yaml(tmp_path / "filters.yaml", {})

        config = load_config(config_dir=tmp_path)

        assert config.sources.substack == []
        assert config.sources.youtube_channels == []
        assert config.sources.hackernews is None
        assert config.sources.arxiv is None

    def test_empty_filters_uses_defaults(self, tmp_path: Path) -> None:
        """Empty filters.yaml results in all sources having default filter state."""
        _write_yaml(tmp_path / "sources.yaml", {})
        _write_yaml(tmp_path / "settings.yaml", _minimal_settings())
        _write_yaml(tmp_path / "filters.yaml", {})

        config = load_config(config_dir=tmp_path)

        # Default: hackernews enabled, substack/youtube/rss disabled
        assert config.filters.hackernews.enabled is True
        assert config.filters.substack.enabled is False
        assert config.filters.youtube.enabled is False
        assert config.filters.rss.enabled is False
        assert config.filters.devto.enabled is True

    def test_partial_settings_uses_defaults(self, tmp_path: Path) -> None:
        """Settings with only required llm block uses defaults for everything else."""
        _write_yaml(tmp_path / "sources.yaml", {})
        _write_yaml(tmp_path / "settings.yaml", _minimal_settings())
        _write_yaml(tmp_path / "filters.yaml", {})

        config = load_config(config_dir=tmp_path)

        assert config.settings.search.default_limit == 20
        assert config.settings.search.hybrid_weight_semantic == 0.6
        assert config.settings.embedding.provider == "openai"

    def test_subreddit_sort_defaults_to_hot(self, tmp_path: Path) -> None:
        """SubredditSource with missing sort defaults to 'hot'."""
        sources = {"subreddits": [{"name": "MachineLearning"}]}
        settings = _minimal_settings()
        _write_yaml(tmp_path / "sources.yaml", sources)
        _write_yaml(tmp_path / "settings.yaml", settings)
        _write_yaml(tmp_path / "filters.yaml", {})

        config = load_config(config_dir=tmp_path)
        sub = config.sources.subreddits[0]
        assert sub.name == "MachineLearning"
        assert sub.sort == "hot"
        assert sub.limit == 25


# ---------------------------------------------------------------------------
# Test: environment variable interpolation
# ---------------------------------------------------------------------------

class TestEnvVarInterpolation:
    """${VAR_NAME} tokens are replaced with environment values at load time."""

    def test_present_env_var_is_substituted(self) -> None:
        """A set environment variable is correctly substituted into the value."""
        raw = {"key": "${MY_TEST_VAR}"}
        with patch.dict(os.environ, {"MY_TEST_VAR": "hello-world"}):
            result = _interpolate_env_vars(raw)
        assert result == {"key": "hello-world"}

    def test_missing_env_var_becomes_empty_string(self, caplog: pytest.LogCaptureFixture) -> None:
        """An unset variable becomes '' and a debug message is logged."""
        raw = {"key": "${DOES_NOT_EXIST_XYZ}"}
        # Ensure the var is definitely absent
        env_copy = {k: v for k, v in os.environ.items() if k != "DOES_NOT_EXIST_XYZ"}
        with patch.dict(os.environ, env_copy, clear=True):
            with caplog.at_level("DEBUG"):
                result = _interpolate_env_vars(raw)
        assert result == {"key": ""}
        assert "DOES_NOT_EXIST_XYZ" in caplog.text

    def test_nested_dict_interpolation(self) -> None:
        """Interpolation descends into nested dicts."""
        raw = {"outer": {"inner": "${NESTED_VAR}"}}
        with patch.dict(os.environ, {"NESTED_VAR": "deep-value"}):
            result = _interpolate_env_vars(raw)
        assert result["outer"]["inner"] == "deep-value"

    def test_list_interpolation(self) -> None:
        """Interpolation descends into list elements."""
        raw = ["${ITEM_ONE}", "${ITEM_TWO}"]
        with patch.dict(os.environ, {"ITEM_ONE": "a", "ITEM_TWO": "b"}):
            result = _interpolate_env_vars(raw)
        assert result == ["a", "b"]

    def test_non_string_scalar_unchanged(self) -> None:
        """Integers, floats, booleans, and None pass through unchanged."""
        raw = {"count": 42, "ratio": 0.5, "flag": True, "nothing": None}
        result = _interpolate_env_vars(raw)
        assert result == raw

    def test_provider_api_key_from_env(self, tmp_path: Path) -> None:
        """API key ${VAR} in settings.yaml resolves to the env value."""
        settings = _minimal_settings()
        settings["providers"] = {"openai": {"api_key": "${OPENAI_API_KEY}"}}
        _write_yaml(tmp_path / "sources.yaml", {})
        _write_yaml(tmp_path / "settings.yaml", settings)
        _write_yaml(tmp_path / "filters.yaml", {})

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-123"}):
            config = load_config(config_dir=tmp_path)

        assert config.settings.providers["openai"].api_key == "sk-test-key-123"

    def test_missing_api_key_env_var_gives_none(self, tmp_path: Path) -> None:
        """A missing API key env var results in None (empty string -> None via loader)."""
        settings = _minimal_settings()
        settings["providers"] = {"openai": {"api_key": "${OPENAI_API_KEY_MISSING}"}}
        _write_yaml(tmp_path / "sources.yaml", {})
        _write_yaml(tmp_path / "settings.yaml", settings)
        _write_yaml(tmp_path / "filters.yaml", {})

        env_copy = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY_MISSING"}
        with patch.dict(os.environ, env_copy, clear=True):
            config = load_config(config_dir=tmp_path)

        key = get_provider_api_key(config, "openai")
        assert key is None


# ---------------------------------------------------------------------------
# Test: ValidationError on bad data
# ---------------------------------------------------------------------------

class TestValidationErrors:
    """Pydantic raises ValidationError on type mismatches and illegal values."""

    def test_invalid_subreddit_sort(self, tmp_path: Path) -> None:
        """sort must be one of hot/new/top/rising."""
        sources = {"subreddits": [{"name": "foo", "sort": "invalid_sort"}]}
        settings = _minimal_settings()
        _write_yaml(tmp_path / "sources.yaml", sources)
        _write_yaml(tmp_path / "settings.yaml", settings)
        _write_yaml(tmp_path / "filters.yaml", {})

        with pytest.raises(ValidationError):
            load_config(config_dir=tmp_path)

    def test_invalid_hackernews_mode(self, tmp_path: Path) -> None:
        """HackerNews mode must be top/new/best."""
        sources = {"hackernews": {"mode": "bogus", "limit": 30}}
        settings = _minimal_settings()
        _write_yaml(tmp_path / "sources.yaml", sources)
        _write_yaml(tmp_path / "settings.yaml", settings)
        _write_yaml(tmp_path / "filters.yaml", {})

        with pytest.raises(ValidationError):
            load_config(config_dir=tmp_path)

    def test_invalid_filter_strategy(self, tmp_path: Path) -> None:
        """Filter strategy must be llm/hybrid/keyword."""
        filters = {"hackernews": {"enabled": True, "strategy": "magic"}}
        _write_yaml(tmp_path / "sources.yaml", {})
        _write_yaml(tmp_path / "settings.yaml", _minimal_settings())
        _write_yaml(tmp_path / "filters.yaml", filters)

        with pytest.raises(ValidationError):
            load_config(config_dir=tmp_path)

    def test_missing_llm_field_defaults_to_none(self, tmp_path: Path) -> None:
        """SettingsConfig allows missing llm block (defaults to None)."""
        _write_yaml(tmp_path / "sources.yaml", {})
        _write_yaml(tmp_path / "settings.yaml", {})
        _write_yaml(tmp_path / "filters.yaml", {})

        config = load_config(config_dir=tmp_path)
        assert config.settings.llm is None

    def test_wrong_type_for_limit(self, tmp_path: Path) -> None:
        """Passing a string for an int field raises ValidationError."""
        sources = {"hackernews": {"mode": "top", "limit": "not-an-int"}}
        settings = _minimal_settings()
        _write_yaml(tmp_path / "sources.yaml", sources)
        _write_yaml(tmp_path / "settings.yaml", settings)
        _write_yaml(tmp_path / "filters.yaml", {})

        with pytest.raises(ValidationError):
            load_config(config_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test: get_provider_api_key
# ---------------------------------------------------------------------------

class TestGetProviderApiKey:
    """get_provider_api_key resolves keys from providers dict."""

    def _make_config(self, providers: dict) -> AppConfig:
        """Helper: build AppConfig with the given providers dict."""
        return AppConfig(
            sources=SourcesConfig(),
            settings=SettingsConfig(
                providers=providers,
                llm=LLMRoutingConfig(
                    filtering=LLMTaskConfig(provider="openrouter", model="llama"),
                    entity_extraction=LLMTaskConfig(provider="openrouter", model="llama"),
                    briefing=LLMTaskConfig(provider="anthropic", model="claude"),
                    source_discovery=LLMTaskConfig(provider="openrouter", model="llama"),
                    keyword_extraction=LLMTaskConfig(provider="openrouter", model="llama"),
                ),
            ),
            filters=FiltersConfig(),
        )

    def test_returns_api_key_for_known_provider(self) -> None:
        """Returns the API key when the provider exists and key is set."""
        from ai_craftsman_kb.config.models import ProviderConfig

        config = self._make_config({"openai": ProviderConfig(api_key="sk-abc123")})
        assert get_provider_api_key(config, "openai") == "sk-abc123"

    def test_returns_none_for_unknown_provider(self) -> None:
        """Returns None when the provider is not in the providers dict."""
        config = self._make_config({})
        assert get_provider_api_key(config, "nonexistent") is None

    def test_returns_none_for_empty_api_key(self) -> None:
        """Returns None when api_key is an empty string (unset env var)."""
        from ai_craftsman_kb.config.models import ProviderConfig

        config = self._make_config({"openai": ProviderConfig(api_key="")})
        assert get_provider_api_key(config, "openai") is None

    def test_returns_none_when_api_key_is_none(self) -> None:
        """Returns None when api_key field is explicitly None."""
        from ai_craftsman_kb.config.models import ProviderConfig

        config = self._make_config({"openai": ProviderConfig(api_key=None)})
        assert get_provider_api_key(config, "openai") is None

    def test_base_url_provider_without_api_key(self) -> None:
        """Providers like Ollama have base_url but no api_key — returns None."""
        from ai_craftsman_kb.config.models import ProviderConfig

        config = self._make_config(
            {"ollama": ProviderConfig(base_url="http://localhost:11434")}
        )
        assert get_provider_api_key(config, "ollama") is None


# ---------------------------------------------------------------------------
# Test: config lookup order
# ---------------------------------------------------------------------------

class TestConfigLookupOrder:
    """Explicit config_dir overrides ~/.ai-craftsman-kb/ which overrides bundled."""

    def test_explicit_dir_takes_priority(self, tmp_path: Path) -> None:
        """An explicit config_dir is used even if ~/.ai-craftsman-kb exists."""
        # Write a recognisable value in the explicit dir
        settings = _minimal_settings()
        settings["server"] = {"backend_port": 9999, "dashboard_port": 4000}
        _write_yaml(tmp_path / "sources.yaml", {})
        _write_yaml(tmp_path / "settings.yaml", settings)
        _write_yaml(tmp_path / "filters.yaml", {})

        config = load_config(config_dir=tmp_path)
        assert config.settings.server.backend_port == 9999

    def test_missing_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """_load_yaml returns {} without raising if the file does not exist."""
        result = _load_yaml(tmp_path / "does_not_exist.yaml")
        assert result == {}

    def test_nonexistent_explicit_dir_falls_back(self, tmp_path: Path) -> None:
        """If explicit dir doesn't exist, falls back to next in lookup chain."""
        non_existent = tmp_path / "no_such_dir"
        # Should not raise; will fall back to ~/.ai-craftsman-kb or bundled
        # (both may not have full configs, but load_config with missing llm raises
        # ValidationError from pydantic, not FileNotFoundError)
        # We just verify the code doesn't crash with a FileNotFoundError
        try:
            load_config(config_dir=non_existent)
        except ValidationError:
            pass  # Expected if fallback dir also lacks required llm config
        except Exception as exc:
            pytest.fail(f"Unexpected exception type: {type(exc).__name__}: {exc}")
