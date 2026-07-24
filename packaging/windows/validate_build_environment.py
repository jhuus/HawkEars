"""Validate native dependencies before starting the expensive Nuitka build."""

import platform

import torch
import torchaudio
import torchvision


def main() -> None:
    expected_versions = {
        "torch": ("2.8.0", torch.__version__),
        "torchvision": ("0.23.0", torchvision.__version__),
        "torchaudio": ("2.8.0", torchaudio.__version__),
    }
    for package, (expected, installed) in expected_versions.items():
        if not installed.startswith(expected):
            raise RuntimeError(f"Expected {package} {expected}, but found {installed}.")
    if torch.version.cuda != "12.6":
        raise RuntimeError(
            f"Expected the CUDA 12.6 PyTorch build, but found {torch.version.cuda}."
        )
    if platform.architecture()[0] != "64bit":
        raise RuntimeError("The HawkEars Windows build requires 64-bit Python.")

    print(
        f"Building with torch {torch.__version__}, "
        f"torchvision {torchvision.__version__}, "
        f"torchaudio {torchaudio.__version__}, "
        f"CUDA runtime {torch.version.cuda}"
    )


if __name__ == "__main__":
    main()
