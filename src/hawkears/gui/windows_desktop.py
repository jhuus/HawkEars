"""Windows taskbar integration for the desktop application."""

import ctypes
from ctypes import wintypes
from importlib.resources import files
from pathlib import Path
import sys
import uuid


APP_USER_MODEL_ID = "HawkEars.HawkEars"


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "_GUID":
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wintypes.DWORD)]


class _PROPVARIANT_VALUE(ctypes.Union):
    _fields_ = [("string", ctypes.c_wchar_p)]


class _PROPVARIANT(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("vt", wintypes.USHORT),
        ("reserved1", wintypes.USHORT),
        ("reserved2", wintypes.USHORT),
        ("reserved3", wintypes.USHORT),
        ("value", _PROPVARIANT_VALUE),
    ]


_IID_PROPERTY_STORE = _GUID.from_string("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")
_APP_USER_MODEL_FORMAT = "9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"
_APP_USER_MODEL_ID_KEY = _PROPERTYKEY(
    _GUID.from_string(_APP_USER_MODEL_FORMAT), 5
)
_RELAUNCH_COMMAND_KEY = _PROPERTYKEY(
    _GUID.from_string(_APP_USER_MODEL_FORMAT), 2
)
_RELAUNCH_ICON_KEY = _PROPERTYKEY(
    _GUID.from_string(_APP_USER_MODEL_FORMAT), 3
)
_VT_LPWSTR = 31


def set_windows_app_user_model_id() -> None:
    """Give CLI-launched GUI windows the HawkEars taskbar identity."""
    if sys.platform != "win32":
        return
    shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
    shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)


def configure_windows_taskbar(window_id: int, launch_command: str) -> None:
    """Set the icon and relaunch metadata on a native HawkEars window."""
    if sys.platform != "win32":
        return

    property_store = ctypes.c_void_p()
    shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
    get_store = shell32.SHGetPropertyStoreForWindow
    get_store.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    get_store.restype = ctypes.HRESULT
    result = get_store(
        wintypes.HWND(window_id),
        ctypes.byref(_IID_PROPERTY_STORE),
        ctypes.byref(property_store),
    )
    if result < 0:
        raise OSError(f"Could not access Windows taskbar properties: {result:#x}")

    icon_resource = f"{windows_icon_path()},0"
    try:
        _set_property(property_store, _APP_USER_MODEL_ID_KEY, APP_USER_MODEL_ID)
        _set_property(property_store, _RELAUNCH_COMMAND_KEY, launch_command)
        _set_property(property_store, _RELAUNCH_ICON_KEY, icon_resource)
        _commit_properties(property_store)
    finally:
        _release_property_store(property_store)


def windows_icon_path() -> Path:
    """Return the ICO used by Windows taskbar and shortcut integration."""
    packaged = Path(str(files("hawkears.gui.ui.resources").joinpath("hawkears.ico")))
    if packaged.is_file():
        return packaged
    source_tree = Path(__file__).resolve().parents[3]
    return source_tree / "packaging" / "windows" / "assets" / "hawkears.ico"


def _property_store_method(store: ctypes.c_void_p, index: int, *argument_types):
    vtable = ctypes.cast(
        store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argument_types)(
        vtable[index]
    )


def _set_property(store: ctypes.c_void_p, key: _PROPERTYKEY, value: str) -> None:
    variant = _PROPVARIANT(vt=_VT_LPWSTR)
    variant.string = value
    method = _property_store_method(
        store, 6, ctypes.POINTER(_PROPERTYKEY), ctypes.POINTER(_PROPVARIANT)
    )
    result = method(store, ctypes.byref(key), ctypes.byref(variant))
    if result < 0:
        raise OSError(f"Could not set Windows taskbar property: {result:#x}")


def _commit_properties(store: ctypes.c_void_p) -> None:
    result = _property_store_method(store, 7)(store)
    if result < 0:
        raise OSError(f"Could not commit Windows taskbar properties: {result:#x}")


def _release_property_store(store: ctypes.c_void_p) -> None:
    _property_store_method(store, 2)(store)
