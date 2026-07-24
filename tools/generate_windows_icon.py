"""Generate the multi-resolution Windows application icon."""

from pathlib import Path
import subprocess
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "hawkears" / "gui" / "ui" / "resources" / "hawkears-icon.svg"
DESTINATION = ROOT / "packaging" / "windows" / "assets" / "hawkears.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DESTINATION.parent) as temporary_directory:
        rendered = Path(temporary_directory) / "hawkears-1024.png"
        subprocess.run(
            [
                "inkscape",
                str(SOURCE),
                "--export-type=png",
                f"--export-filename={rendered}",
                "--export-width=1024",
                "--export-height=1024",
            ],
            check=True,
        )
        with Image.open(rendered) as image:
            image.convert("RGBA").save(
                DESTINATION,
                format="ICO",
                sizes=[(size, size) for size in SIZES],
            )
    print(DESTINATION)


if __name__ == "__main__":
    main()
