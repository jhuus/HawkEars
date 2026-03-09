#!/usr/bin/env python3

from .__about__ import __version__

__all__ = ["__version__"]

# SPDX-FileCopyrightText: 2025-present Jan Huus <jhuus1@gmail.com>
#
# SPDX-License-Identifier: MIT

# This lets you do "from hawkears import __version__" anywhere
try:
    from .__about__ import __version__  # type: ignore
except Exception:
    try:
        from importlib.metadata import version as _pkg_version  # type: ignore

        __version__ = _pkg_version("hawkears")  # type: ignore[assignment]
    except Exception:
        __version__ = "0.0.0"

from hawkears import commands
from hawkears.core.config_loader import get_config

__all__ = [
    "__version__",
    "commands",
    "get_config",
]
