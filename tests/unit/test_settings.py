from pathlib import Path

import pytest
from pydantic import ValidationError

from invariant.config.settings import (
    ConfigError,
    find_project_root,
    load_config,
    load_settings,
)

ROOT = Path(__file__).resolve().parents[2]
ENV_VARS = ("FRED_API_KEY", "INVARIANT_DATA_DIR", "INVARIANT_RUNS_DIR")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_env_file(tmp_path: Path) -> Path:
    return tmp_path / "does-not-exist.env"


def test_find_project_root_is_repo_root() -> None:
    assert find_project_root() == ROOT
    assert (ROOT / "pyproject.toml").is_file()


def test_find_project_root_from_nested_start(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path.resolve()
    assert find_project_root(nested / "module.py") == tmp_path.resolve()


def test_find_project_root_missing_raises(tmp_path: Path) -> None:
    if any((p / "pyproject.toml").is_file() for p in [tmp_path, *tmp_path.parents]):
        pytest.skip("tmp dir has a pyproject.toml ancestor")
    with pytest.raises(ConfigError, match=r"pyproject\.toml"):
        find_project_root(tmp_path)


def test_defaults_resolve_against_root(no_env_file: Path) -> None:
    settings = load_settings(no_env_file)
    assert settings.fred_api_key is None
    assert settings.data_dir == ROOT / "data"
    assert settings.runs_dir == ROOT / "runs"


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch, no_env_file: Path) -> None:
    monkeypatch.setenv("FRED_API_KEY", "secret-key")
    monkeypatch.setenv("INVARIANT_DATA_DIR", "elsewhere/data")
    settings = load_settings(no_env_file)
    assert settings.fred_api_key is not None
    assert settings.fred_api_key.get_secret_value() == "secret-key"
    assert settings.data_dir == ROOT / "elsewhere" / "data"


def test_absolute_env_path_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_env_file: Path
) -> None:
    monkeypatch.setenv("INVARIANT_RUNS_DIR", str(tmp_path / "runs"))
    assert load_settings(no_env_file).runs_dir == tmp_path / "runs"


def test_env_file_values_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text("FRED_API_KEY=from-file\nINVARIANT_DATA_DIR=file_data\nUNRELATED=1\n")
    settings = load_settings(env_file)
    assert settings.fred_api_key is not None
    assert settings.fred_api_key.get_secret_value() == "from-file"
    assert settings.data_dir == ROOT / "file_data"


def test_empty_key_in_env_file_is_none(tmp_path: Path) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text("FRED_API_KEY=\n")
    assert load_settings(env_file).fred_api_key is None


def test_secret_not_exposed(monkeypatch: pytest.MonkeyPatch, no_env_file: Path) -> None:
    monkeypatch.setenv("FRED_API_KEY", "super-secret-value")
    settings = load_settings(no_env_file)
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.model_dump())
    assert "super-secret-value" not in settings.model_dump_json()


def test_settings_frozen(no_env_file: Path) -> None:
    settings = load_settings(no_env_file)
    with pytest.raises(ValidationError):
        settings.data_dir = Path("/tmp/x")  # type: ignore[misc]


class TestIndependentOfCwd:
    def test_relative_env_var_resolves_against_root_not_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("INVARIANT_DATA_DIR", "foo")
        settings = load_settings()
        assert settings.data_dir == ROOT / "foo"
        assert settings.data_dir != tmp_path / "foo"

    def test_default_env_file_is_root_not_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from_root = load_settings()
        (tmp_path / ".env").write_text("INVARIANT_RUNS_DIR=cwd_runs\nFRED_API_KEY=cwd-key\n")
        monkeypatch.chdir(tmp_path)
        from_tmp = load_settings()
        assert from_tmp == from_root
        assert from_tmp.runs_dir != ROOT / "cwd_runs"

    def test_defaults_with_foreign_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_env_file: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        settings = load_settings(no_env_file)
        assert settings.data_dir == ROOT / "data"
        assert settings.runs_dir == ROOT / "runs"

    def test_load_config_with_foreign_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        expected = load_config()
        monkeypatch.chdir(tmp_path)
        assert load_config() == expected
        assert load_config(Path("configs/base.yaml")) == expected
