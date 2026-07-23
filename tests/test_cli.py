from click.testing import CliRunner

from hawkears.cli import cli


def test_gui_command_launches_desktop_entry_point(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hawkears.gui.app.main", lambda argv=None: calls.append(argv) or 0
    )

    result = CliRunner().invoke(cli, ["gui"])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert len(calls[0]) == 1


def test_gui_command_is_listed_in_help():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "gui" in result.output
    assert "Launch the HawkEars desktop GUI." in result.output
