from pathlib import Path

from hawkears.core.app_paths import (
    default_data_directory,
    is_application_ready,
    resolve_application_paths,
)
from hawkears.core import config_loader
from hawkears.core.initializer import get_initialization_status, is_initialized


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
    assert (
        Path(config.hawkears.low_band_ckpt_folder)
        == tmp_path / "data" / "ckpt-low-band"
    )
    assert Path(config.hawkears.exclude_list) == tmp_path / "data" / "exclude.txt"
    assert Path(config.hawkears.taxonomy_file).name == "taxonomy.csv"
    assert Path(config.hawkears.taxonomy_file).is_file()
    assert (
        Path(config.hawkears.occurrence_pickle) == tmp_path / "data" / "occurrence.pkl"
    )


def test_local_taxonomy_overrides_packaged_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config_loader.util, "get_device", lambda: "cpu")
    config_loader._base_configs.clear()
    taxonomy = tmp_path / "data" / "taxonomy.csv"
    taxonomy.parent.mkdir()
    taxonomy.write_text("model_code,name,code,alt_name,alt_code\n", encoding="utf-8")

    config = config_loader.get_config(data_root=tmp_path)

    assert Path(config.hawkears.taxonomy_file) == taxonomy


def test_application_readiness_requires_catalogs_and_both_model_sets(tmp_path):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml" / "default.yaml").touch()
    data = tmp_path / "data"
    data.mkdir()
    (data / "classes.csv").touch()
    (data / "locations.db").touch()
    (data / "ckpt").mkdir()
    (data / "ckpt" / "main.ckpt").touch()

    assert not is_application_ready(tmp_path)

    (data / "ckpt-low-band").mkdir()
    (data / "ckpt-low-band" / "low.onnx").touch()

    assert is_application_ready(tmp_path)
    assert is_initialized(tmp_path)


def test_initialization_status_reports_missing_and_outdated_bundles(tmp_path):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml" / "default.yaml").touch()
    data = tmp_path / "data"
    data.mkdir()
    (data / "classes.csv").touch()
    (data / "locations.db").touch()
    (data / "ckpt").mkdir()
    (data / "ckpt" / "main.ckpt").touch()
    (data / "models.json").write_text(
        '{"format_version": 2, "package_region": "canada", '
        '"bundles": {"main": {"version": "old", "path": "data/ckpt"}}}',
        encoding="utf-8",
    )

    status = get_initialization_status(tmp_path)

    assert not status.ready
    assert status.missing_bundles == ("low_band",)
    assert status.outdated_bundles == ("main",)


def test_version_one_manifest_uses_default_bundle_paths(tmp_path):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml" / "default.yaml").touch()
    data = tmp_path / "data"
    data.mkdir()
    (data / "classes.csv").touch()
    (data / "locations.db").touch()
    for name in ("ckpt", "ckpt-low-band"):
        directory = data / name
        directory.mkdir()
        (directory / "model.ckpt").touch()
    (data / "models.json").write_text(
        '{"format_version": 1, "bundles": {'
        '"main": {"version": "2.2.0"}, '
        '"low_band": {"version": "2.0.0"}}}',
        encoding="utf-8",
    )

    assert get_initialization_status(tmp_path).ready


def test_initialization_status_uses_configured_checkpoint_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config_loader.util, "get_device", lambda: "cpu")
    config_loader._base_configs.clear()
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml" / "default.yaml").write_text(
        "misc:\n  ckpt_folder: custom/main\n"
        "hawkears:\n  low_band_ckpt_folder: custom/low\n",
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / "classes.csv").touch()
    (data / "locations.db").touch()
    for name in ("main", "low"):
        directory = tmp_path / "custom" / name
        directory.mkdir(parents=True)
        (directory / "model.ckpt").touch()

    assert get_initialization_status(tmp_path).ready
