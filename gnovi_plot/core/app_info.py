"""Single authoritative source for Gnovi Studio's product identity.

`__version__` must stay in sync with `pyproject.toml`'s `[project].version`
(the packaging build reads the latter; the app reads this module) --
`tests/test_app_info.py` guards against the two drifting apart.
"""

from __future__ import annotations

APP_NAME = "Gnovi Studio"
APP_TAGLINE = "Scientific Plotting & Visualization"

__version__ = "0.9.0"
VERSION_LABEL = "v0.9.0 Beta"

REPO_URL = "https://github.com/wavicles/Gnovi-Plot"
ISSUES_URL = f"{REPO_URL}/issues"


def about_text() -> str:
    """Body text for the About dialog -- kept as a plain function so it's
    testable without introspecting a live QMessageBox."""
    return f"{APP_NAME}\n{APP_TAGLINE}\n\n{VERSION_LABEL}\n{REPO_URL}"
