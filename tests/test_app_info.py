import re
from pathlib import Path

from gnovi_plot.core.app_info import APP_NAME, APP_TAGLINE, VERSION_LABEL, __version__, about_text

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_identity_constants_are_nonempty():
    assert APP_NAME == "Gnovi Studio"
    assert APP_TAGLINE
    assert __version__
    assert VERSION_LABEL


def test_pyproject_version_matches_app_info_version():
    """Guards against `pyproject.toml`'s `[project].version` (read by the
    packaging build) drifting from `gnovi_plot.core.app_info.__version__`
    (read by the running app) -- they must always agree. Parsed with a
    targeted regex rather than `tomllib` so this test doesn't impose a
    Python 3.11+ floor beyond what the rest of the suite requires.
    """
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', _PYPROJECT_PATH.read_text())
    assert match is not None, "pyproject.toml has no [project].version"
    assert match.group(1) == __version__


def test_about_text_includes_name_version_and_tagline():
    text = about_text()
    assert APP_NAME in text
    assert APP_TAGLINE in text
    assert VERSION_LABEL in text
