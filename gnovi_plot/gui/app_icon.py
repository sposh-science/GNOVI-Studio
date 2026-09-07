"""The GNOVI Studio application icon, loaded from resources shipped inside
the package (``gnovi_plot/resources/icons/``) so it resolves from any
install -- wheel, editable checkout, or distro package -- without a
filesystem path relative to this source file.

Raster PNGs are used for the runtime :class:`QIcon`, not the SVG, so the
window/taskbar icon does not depend on the optional Qt SVG plugin being
installed (Debian, for one, splits it into a separate ``python3-pyside6``
subpackage). The scalable ``gnovi-studio.svg`` in the same directory is
kept for downstream packagers who install into ``hicolor/scalable``.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from PySide6.QtGui import QIcon, QPixmap

#: Icon-name shared by the runtime icon, the ``.desktop`` ``Icon=`` key and
#: ``QApplication.setDesktopFileName`` -- keep the three in agreement.
ICON_NAME = "gnovi-studio"

_ICON_DIR = files("gnovi_plot") / "resources" / "icons"
_PNG_SIZES = (48, 128, 256, 512)


@lru_cache(maxsize=1)
def app_icon() -> QIcon:
    """The application icon as a multi-resolution :class:`QIcon`.

    Assembled from the bundled PNG renders; the result is cached, so this
    is cheap to call repeatedly. Requires a running ``QApplication``
    (``QPixmap`` does).
    """
    icon = QIcon()
    for size in _PNG_SIZES:
        pixmap = QPixmap()
        pixmap.loadFromData((_ICON_DIR / f"{ICON_NAME}-{size}.png").read_bytes(), "PNG")
        if not pixmap.isNull():
            icon.addPixmap(pixmap)
    return icon
