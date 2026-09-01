from hawkears.gui import linux_desktop


def test_desktop_integration_is_a_noop_outside_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(linux_desktop.sys, "platform", "win32")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert not linux_desktop.install_linux_desktop_integration()
    assert not any(tmp_path.iterdir())


def test_desktop_integration_installs_linux_launcher_and_icon(monkeypatch, tmp_path):
    source_icon = tmp_path / "source.svg"
    source_icon.write_text("<svg/>", encoding="utf-8")
    data_home = tmp_path / "share"
    monkeypatch.setattr(linux_desktop.sys, "platform", "linux")
    monkeypatch.setattr(linux_desktop, "brand_icon_path", lambda: str(source_icon))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    assert linux_desktop.install_linux_desktop_integration()
    launcher = data_home / "applications" / linux_desktop.DESKTOP_FILE_NAME
    icon = data_home / "icons" / "hicolor" / "scalable" / "apps" / "hawkears.svg"
    assert "Exec=hawkears-gui %f" in launcher.read_text(encoding="utf-8")
    assert icon.read_text(encoding="utf-8") == "<svg/>"
    assert not linux_desktop.install_linux_desktop_integration()
