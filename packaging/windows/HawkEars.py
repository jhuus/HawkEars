"""Nuitka launcher for the HawkEars desktop application."""


def _disable_numba_decorator_caching() -> None:
    """Avoid caches that require Python sources omitted from the bundle."""
    import numba

    for decorator_name in ("guvectorize", "jit", "njit", "vectorize"):
        decorator = getattr(numba, decorator_name)

        def without_cache(*args, _decorator=decorator, **kwargs):
            kwargs["cache"] = False
            return _decorator(*args, **kwargs)

        setattr(numba, decorator_name, without_cache)


_disable_numba_decorator_caching()

from hawkears.gui.app import main

raise SystemExit(main())
