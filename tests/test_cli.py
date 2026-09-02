from click.testing import CliRunner
import signal

from hawkears.cli import cli
from hawkears.commands._analyze import _validate_label_length_limits
from hawkears.gui.app import _initial_project_path


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


def test_gui_accepts_an_existing_project_from_file_association(tmp_path):
    project = tmp_path / "survey.hawkears"
    project.touch()

    assert _initial_project_path([str(project)]) == project.resolve()
    assert _initial_project_path(["--unexpected"]) is None


def test_analyze_ctrl_c_requests_cooperative_cancellation(tmp_path, monkeypatch):
    installed_handlers = []
    previous_handler = object()

    monkeypatch.setattr(signal, "getsignal", lambda signum: previous_handler)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed_handlers.append(handler),
    )

    def fake_analyze(*args, **kwargs):
        assert not kwargs["cancellation_callback"]()
        installed_handlers[-1](signal.SIGINT, None)
        assert kwargs["cancellation_callback"]()

    monkeypatch.setattr("hawkears.commands._analyze.analyze", fake_analyze)

    result = CliRunner().invoke(cli, ["analyze", str(tmp_path), "--quiet"])

    assert result.exit_code == 1
    assert "Cancellation requested; finishing active recordings." in result.output
    assert "Aborted!" in result.output
    assert installed_handlers[-1] is previous_handler


def test_analyze_second_ctrl_c_aborts_immediately(tmp_path, monkeypatch):
    installed_handlers = []
    previous_handler = object()

    monkeypatch.setattr(signal, "getsignal", lambda signum: previous_handler)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed_handlers.append(handler),
    )

    def fake_analyze(*args, **kwargs):
        installed_handlers[-1](signal.SIGINT, None)
        installed_handlers[-1](signal.SIGINT, None)

    monkeypatch.setattr("hawkears.commands._analyze.analyze", fake_analyze)

    result = CliRunner().invoke(cli, ["analyze", str(tmp_path), "--quiet"])

    assert result.exit_code == 1
    assert "Aborted!" in result.output
    assert installed_handlers[-1] is previous_handler


def test_analyze_forwards_minimum_label_length(tmp_path, monkeypatch):
    received = {}

    def fake_analyze(*args, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr("hawkears.commands._analyze.analyze", fake_analyze)

    result = CliRunner().invoke(
        cli,
        ["analyze", str(tmp_path), "--min-label-length", "0.5", "--quiet"],
    )

    assert result.exit_code == 0
    assert received["min_label_length"] == 0.5


def test_minimum_label_length_validation():
    from types import SimpleNamespace

    def config(minimum, maximum=None, segment_len=None):
        return SimpleNamespace(
            hawkears=SimpleNamespace(
                min_label_length=minimum,
                max_label_length=maximum,
                max_label_length_merge_threshold=0.5,
            ),
            infer=SimpleNamespace(segment_len=segment_len),
        )

    _validate_label_length_limits(config(0.25))
    _validate_label_length_limits(config(0.5, maximum=1.0))

    for cfg, message in (
        (config(0), "greater than zero"),
        (config(0.3), "multiple of 0.25"),
        (config(0.5, segment_len=3.0), "variable-length labels"),
        (config(1.25, maximum=1.0), "cannot exceed"),
    ):
        try:
            _validate_label_length_limits(cfg)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("Expected invalid label length configuration")
