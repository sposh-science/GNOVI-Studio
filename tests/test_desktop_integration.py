"""Linux desktop integration: the bundled icon resources, the runtime
QIcon, and the canonical `packaging/linux/gnovi-studio.desktop` entry.

See `packaging/linux/README.md` for how downstream packagers consume these.
"""

from __future__ import annotations

import configparser
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest

from gnovi_plot.gui.app_icon import ICON_NAME, app_icon

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DESKTOP_FILE = _REPO_ROOT / "packaging" / "linux" / "gnovi-studio.desktop"
_ICON_SIZES = (48, 128, 256, 512)


def _desktop_entry() -> configparser.SectionProxy:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # desktop-entry keys are case-sensitive
    parser.read(_DESKTOP_FILE, encoding="utf-8")
    return parser["Desktop Entry"]


# --- bundled resources ------------------------------------------------------


def test_icon_resources_are_shipped_in_the_package():
    icon_dir = files("gnovi_plot") / "resources" / "icons"
    assert (icon_dir / f"{ICON_NAME}.svg").is_file()
    for size in _ICON_SIZES:
        assert (icon_dir / f"{ICON_NAME}-{size}.png").is_file()


def test_app_icon_builds_a_non_null_multi_size_icon(qapp):
    icon = app_icon()
    assert not icon.isNull()
    # every bundled PNG size made it in
    assert {s.width() for s in icon.availableSizes()} >= set(_ICON_SIZES)


def test_app_icon_is_cached(qapp):
    assert app_icon() is app_icon()


# --- .desktop entry -------------------------------------------------------


def test_desktop_file_exists_and_parses():
    assert _DESKTOP_FILE.is_file()
    entry = _desktop_entry()
    assert entry["Type"] == "Application"
    assert entry["Name"] == "GNOVI Studio"


def test_desktop_exec_and_icon_match_the_launcher_and_icon_name():
    entry = _desktop_entry()
    # Exec must be exactly the gui-scripts entry point name, on PATH
    assert entry["Exec"] == "gnovi-studio"
    # Icon is an icon-theme name shared with setDesktopFileName / app_icon
    assert entry["Icon"] == ICON_NAME == "gnovi-studio"
    assert _DESKTOP_FILE.stem == ICON_NAME  # base name drives WM association


def test_desktop_declares_no_mimetype_because_gnovi_opens_no_argv_files():
    assert "MimeType" not in _desktop_entry()


def test_desktop_categories_are_registered_freedesktop_categories():
    cats = [c for c in _desktop_entry()["Categories"].split(";") if c]
    assert cats == ["Science", "Education", "DataVisualization"]


def test_pyproject_declares_the_gnovi_studio_gui_script():
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.gui-scripts]' in text
    assert 'gnovi-studio = "gnovi_plot.app:main"' in text


@pytest.mark.skipif(
    shutil.which("desktop-file-validate") is None,
    reason="desktop-file-validate (desktop-file-utils) not installed",
)
def test_desktop_file_passes_desktop_file_validate():
    result = subprocess.run(
        ["desktop-file-validate", str(_DESKTOP_FILE)],
        capture_output=True,
        text=True,
    )
    # hints (e.g. "more than one main category") are allowed; errors/warnings are not
    assert result.returncode == 0, result.stdout + result.stderr
    assert "error:" not in result.stdout.lower()
    assert "warning:" not in result.stdout.lower()
