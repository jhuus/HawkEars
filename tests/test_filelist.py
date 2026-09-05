from pathlib import Path

import pytest

from hawkears.core.filelist import resolve_filelist_metadata


def test_filelist_supports_unique_basenames_and_relative_paths(tmp_path: Path):
    root = tmp_path / "recordings"
    first = root / "site-a" / "recording.wav"
    second = root / "site-b" / "recording.wav"
    unique = root / "unique.wav"
    absolute = root / "absolute.wav"
    for path in (first, second, unique, absolute):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    filelist = tmp_path / "filelist.csv"
    filelist.write_text(
        "filename,region,recording_date\n"
        "site-a/recording.wav,CA-ON-OT,2026-05-18\n"
        "site-b\\recording.wav,CA-QC-MR,2026-05-19\n"
        "unique.wav,CA-BC-GV,2026-05-20\n"
        f"{absolute},CA-AB-ED,2026-05-21\n",
        encoding="utf-8",
    )

    result = resolve_filelist_metadata(
        filelist, [first, second, unique, absolute], root
    )

    assert result == {
        first.resolve(): {
            "recorded_at": "2026-05-18",
            "region_code": "CA-ON-OT",
        },
        second.resolve(): {
            "recorded_at": "2026-05-19",
            "region_code": "CA-QC-MR",
        },
        unique.resolve(): {
            "recorded_at": "2026-05-20",
            "region_code": "CA-BC-GV",
        },
        absolute.resolve(): {
            "recorded_at": "2026-05-21",
            "region_code": "CA-AB-ED",
        },
    }


@pytest.mark.parametrize("directories", [("site-a", "site-b"), (".", "site-b")])
def test_filelist_rejects_ambiguous_basename(tmp_path: Path, directories):
    root = tmp_path / "recordings"
    paths = [root / directory / "recording.wav" for directory in directories]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    filelist = tmp_path / "filelist.csv"
    filelist.write_text(
        "filename,region,recording_date\n" "recording.wav,CA-ON-OT,2026-05-18\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Ambiguous filename 'recording.wav'") as error:
        resolve_filelist_metadata(filelist, paths, root)

    assert str(paths[0].resolve()) in str(error.value)
    assert str(paths[1].resolve()) in str(error.value)
    assert "relative to the recording directory" in str(error.value)

    filelist.write_text(
        "filename,region,recording_date\n./recording.wav,CA-ON-OT,2026-05-18\n"
    )
    if directories[0] == ".":
        assert list(resolve_filelist_metadata(filelist, paths, root)) == [paths[0]]
