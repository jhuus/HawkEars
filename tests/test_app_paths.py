from pathlib import Path

from hawkears.core.app_paths import (
    default_data_directory,
    resolve_application_paths,
)
from hawkears.core import config_loader


def test_explicit_data_root_takes_precedence(tmp_path):
    explicit = tmp_path / "explicit"
    legacy = tmp_path / "legacy"
    (legacy / "yaml").mkdir(parents=True)
    (legacy / "yaml" / "default.yaml").touch()
    (legacy / "data").mkdir()

    paths = resolve_application_paths(
        explicit,
        cwd=legacy,
        environ={"HAWKEARS_DATA_DIR": str(tmp_path / "configured")},
    )

    assert paths.data_root == explicit


def test_configured_data_root_takes_precedence_over_legacy_directory(tmp_path):
    configured = tmp_path / "configured"
    legacy = tmp_path / "legacy"
    (legacy / "yaml").mkdir(parents=True)
    (legacy / "yaml" / "default.yaml").touch()
    (legacy / "data").mkdir()

    paths = resolve_application_paths(
        cwd=legacy,
        environ={"HAWKEARS_DATA_DIR": str(configured)},
    )

    assert paths.data_root == configured


def test_initialized_current_directory_retains_legacy_behavior(tmp_path):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml" / "default.yaml").touch()
    (tmp_path / "data").mkdir()

    assert resolve_application_paths(cwd=tmp_path, environ={}).data_root == tmp_path


def test_platform_default_data_directories():
    home = Path("/users/birder")

    assert default_data_directory(
        platform="win32",
        home=home,
        environ={"LOCALAPPDATA": "C:/Users/Birder/AppData/Local"},
    ) == Path("C:/Users/Birder/AppData/Local/HawkEars")
    assert (
        default_data_directory(platform="darwin", home=home, environ={})
        == home / "Library" / "Application Support" / "HawkEars"
    )
    assert default_data_directory(
        platform="linux",
        home=home,
        environ={"XDG_DATA_HOME": "/users/birder/.data"},
    ) == Path("/users/birder/.data/hawkears")


def test_packaged_config_uses_explicit_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config_loader.util, "get_device", lambda: "cpu")
    config_loader._base_configs.clear()

    config = config_loader.get_config(data_root=tmp_path)

    assert Path(config.misc.ckpt_folder) == tmp_path / "data" / "ckpt"
    assert Path(config.hawkears.exclude_list) == tmp_path / "data" / "exclude.txt"
    assert (
        Path(config.hawkears.occurrence_pickle) == tmp_path / "data" / "occurrence.pkl"
    )
