from pathlib import Path

from hawkears.gui import linux_desktop


def test_installs_linux_desktop_launcher_and_icon(monkeypatch, tmp_path):
    source_icon = tmp_path / "source.svg"
    source_icon.write_text("<svg/>", encoding="utf-8")
    data_home = tmp_path / "desktop data"
    executable = tmp_path / "Python Environment" / "python"
    monkeypatch.setattr(linux_desktop, "brand_icon_path", lambda: str(source_icon))

    installed = linux_desktop.install_linux_desktop_integration(
        environ={"XDG_DATA_HOME": str(data_home)},
        home=tmp_path,
        executable=executable,
        platform="linux",
    )

    assert installed
    icon = data_home / "icons/hicolor/scalable/apps/hawkears.svg"
    launcher = data_home / "applications/hawkears.desktop"
    assert icon.read_text(encoding="utf-8") == "<svg/>"
    contents = launcher.read_text(encoding="utf-8")
    assert "Name=HawkEars" in contents
    assert (
        'Exec="' + str(executable.absolute()) + '" -m hawkears.gui.app %f' in contents
    )
    assert "Icon=hawkears" in contents
    assert "StartupWMClass=HawkEars" in contents
    assert not linux_desktop.install_linux_desktop_integration(
        environ={"XDG_DATA_HOME": str(data_home)},
        home=tmp_path,
        executable=executable,
        platform="linux",
    )


def test_desktop_integration_is_linux_only(tmp_path):
    assert not linux_desktop.install_linux_desktop_integration(
        environ={"XDG_DATA_HOME": str(tmp_path)},
        home=tmp_path,
        executable=Path("/usr/bin/python3"),
        platform="darwin",
    )
    assert not (tmp_path / "applications").exists()
