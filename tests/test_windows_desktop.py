from types import SimpleNamespace

from hawkears.gui import windows_desktop


def test_windows_app_user_model_id_is_set_on_windows(monkeypatch):
    calls = []
    shell32 = SimpleNamespace(
        SetCurrentProcessExplicitAppUserModelID=calls.append
    )
    monkeypatch.setattr(windows_desktop.sys, "platform", "win32")
    monkeypatch.setattr(
        windows_desktop.ctypes,
        "windll",
        SimpleNamespace(shell32=shell32),
        raising=False,
    )

    windows_desktop.set_windows_app_user_model_id()

    assert calls == [windows_desktop.APP_USER_MODEL_ID]


def test_windows_app_user_model_id_is_skipped_elsewhere(monkeypatch):
    monkeypatch.setattr(windows_desktop.sys, "platform", "linux")
    monkeypatch.delattr(windows_desktop.ctypes, "windll", raising=False)

    windows_desktop.set_windows_app_user_model_id()


def test_windows_taskbar_configuration_is_skipped_elsewhere(monkeypatch):
    monkeypatch.setattr(windows_desktop.sys, "platform", "linux")
    monkeypatch.delattr(windows_desktop.ctypes, "windll", raising=False)

    windows_desktop.configure_windows_taskbar(123, '"hawkears" gui')


def test_windows_icon_is_available_from_source_tree():
    assert windows_desktop.windows_icon_path().is_file()
