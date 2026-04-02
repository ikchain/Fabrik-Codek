"""Tests for dynamic config hot-reload."""

import json

import pytest

from src.core.dynamic_config import get_dynamic, reset_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    """Reset module cache before each test."""
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# TestGetDynamic
# ---------------------------------------------------------------------------


class TestGetDynamic:
    def test_no_env_no_file_returns_default(self):
        assert get_dynamic("nonexistent_key", default=42) == 42

    def test_env_var_highest_priority(self, monkeypatch):
        monkeypatch.setenv("FABRIK_DYNAMIC_MY_THRESHOLD", "0.99")
        assert get_dynamic("my_threshold", default=0.5) == 0.99

    def test_json_file_returns_value(self, tmp_path, monkeypatch):
        config = {"my_key": "from_json"}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("my_key", default="fallback") == "from_json"

    def test_env_overrides_json(self, tmp_path, monkeypatch):
        config = {"score": 0.5}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        monkeypatch.setenv("FABRIK_DYNAMIC_SCORE", "0.99")
        assert get_dynamic("score", default=0.1) == 0.99

    def test_bool_true_variants(self, monkeypatch):
        for val in ("true", "True", "TRUE", "1", "yes", "YES"):
            monkeypatch.setenv("FABRIK_DYNAMIC_FLAG", val)
            assert get_dynamic("flag", default=False) is True, f"Failed for {val!r}"

    def test_bool_false_variants(self, monkeypatch):
        for val in ("false", "False", "0", "no", "NO", "anything"):
            monkeypatch.setenv("FABRIK_DYNAMIC_FLAG", val)
            assert get_dynamic("flag", default=False) is False, f"Failed for {val!r}"

    def test_float_cast_from_env(self, monkeypatch):
        monkeypatch.setenv("FABRIK_DYNAMIC_THRESHOLD", "0.42")
        result = get_dynamic("threshold", default=0.1)
        assert result == 0.42
        assert isinstance(result, float)

    def test_int_cast_from_env(self, monkeypatch):
        monkeypatch.setenv("FABRIK_DYNAMIC_LIMIT", "100")
        result = get_dynamic("limit", default=50)
        assert result == 100
        assert isinstance(result, int)

    def test_malformed_json_returns_default(self, tmp_path, monkeypatch):
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text("{invalid json")
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("any_key", default="safe") == "safe"

    def test_file_deleted_returns_default(self, tmp_path, monkeypatch):
        config_path = tmp_path / "dynamic_config.json"
        # File doesn't exist
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("missing", default=99) == 99


# ---------------------------------------------------------------------------
# TestCastInference
# ---------------------------------------------------------------------------


class TestCastInference:
    def test_float_default_infers_float(self, tmp_path, monkeypatch):
        config = {"val": "0.75"}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        result = get_dynamic("val", default=0.5)
        assert result == 0.75
        assert isinstance(result, float)

    def test_bool_default_infers_bool(self, tmp_path, monkeypatch):
        config = {"enabled": "yes"}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        result = get_dynamic("enabled", default=False)
        assert result is True

    def test_none_default_no_cast_returns_raw(self, monkeypatch):
        monkeypatch.setenv("FABRIK_DYNAMIC_RAW", "hello")
        result = get_dynamic("raw")
        assert result == "hello"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestMtimeCache
# ---------------------------------------------------------------------------


class TestMtimeCache:
    def test_same_mtime_no_reread(self, tmp_path, monkeypatch):
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps({"k": "v1"}))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)

        # First read
        assert get_dynamic("k", default="x") == "v1"

        # Overwrite content but preserve mtime (write same second)
        config_path.write_text(json.dumps({"k": "v2"}))
        # Force same mtime
        import src.core.dynamic_config as dc

        dc._cached_mtime = config_path.stat().st_mtime

        # Should return cached value
        assert get_dynamic("k", default="x") == "v1"

    def test_changed_mtime_rereads(self, tmp_path, monkeypatch):
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps({"k": "v1"}))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)

        # First read
        assert get_dynamic("k", default="x") == "v1"

        # Write new content and force different mtime
        config_path.write_text(json.dumps({"k": "v2"}))
        import src.core.dynamic_config as dc

        dc._cached_mtime = 0.0  # Force cache miss

        assert get_dynamic("k", default="x") == "v2"

    def test_file_not_exists_no_error(self, tmp_path, monkeypatch):
        config_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("any", default="ok") == "ok"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_env_var_treated_as_unset(self, tmp_path, monkeypatch):
        """Empty string env var falls through to JSON/default."""
        config = {"threshold": 0.42}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        monkeypatch.setenv("FABRIK_DYNAMIC_THRESHOLD", "")
        assert get_dynamic("threshold", default=0.1) == 0.42

    def test_json_native_bool_passthrough(self, tmp_path, monkeypatch):
        """Native JSON bool passes through without string casting."""
        config = {"enabled": True}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        result = get_dynamic("enabled", default=False)
        assert result is True

    def test_json_native_float_passthrough(self, tmp_path, monkeypatch):
        """Native JSON float passes through without string casting."""
        config = {"score": 0.85}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        result = get_dynamic("score", default=0.5)
        assert result == 0.85
        assert isinstance(result, float)

    def test_json_native_int_passthrough(self, tmp_path, monkeypatch):
        """Native JSON int passes through without string casting."""
        config = {"limit": 42}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        result = get_dynamic("limit", default=10)
        assert result == 42
        assert isinstance(result, int)

    def test_cast_failure_returns_default(self, monkeypatch):
        """Invalid cast (e.g., 'hello' as int) returns default."""
        monkeypatch.setenv("FABRIK_DYNAMIC_LIMIT", "not_a_number")
        assert get_dynamic("limit", default=50) == 50

    def test_explicit_cast_overrides_inferred(self, monkeypatch):
        """Explicit cast parameter takes precedence over type(default)."""
        monkeypatch.setenv("FABRIK_DYNAMIC_VAL", "42")
        # default is str but cast=int
        result = get_dynamic("val", default="fallback", cast=int)
        assert result == 42
        assert isinstance(result, int)

    def test_json_non_dict_root_returns_default(self, tmp_path, monkeypatch):
        """JSON with array root is rejected."""
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text("[1, 2, 3]")
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("any", default="safe") == "safe"

    def test_json_null_value_falls_to_default(self, tmp_path, monkeypatch):
        """JSON key with null value falls through to default."""
        config = {"key": None}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("key", default="fallback") == "fallback"

    def test_multiple_keys_only_requested_returned(self, tmp_path, monkeypatch):
        """Only the requested key is returned from a multi-key JSON."""
        config = {"a": 1, "b": 2, "c": 3}
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("b", default=0) == 2

    def test_reset_cache_clears_state(self, tmp_path, monkeypatch):
        """reset_cache allows re-initialization."""
        config_path = tmp_path / "dynamic_config.json"
        config_path.write_text(json.dumps({"k": "cached"}))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("k", default="x") == "cached"

        reset_cache()
        # After reset, _initialized is False, path will be re-resolved
        # Point to a new file
        config_path2 = tmp_path / "dynamic_config2.json"
        config_path2.write_text(json.dumps({"k": "fresh"}))
        monkeypatch.setattr("src.core.dynamic_config._config_path", config_path2)
        monkeypatch.setattr("src.core.dynamic_config._initialized", True)
        assert get_dynamic("k", default="x") == "fresh"

    def test_env_var_case_insensitive_name(self, monkeypatch):
        """Name is uppercased for env var lookup."""
        monkeypatch.setenv("FABRIK_DYNAMIC_MY_SETTING", "works")
        assert get_dynamic("my_setting", default="no") == "works"
        assert get_dynamic("MY_SETTING", default="no") == "works"
