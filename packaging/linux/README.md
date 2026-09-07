# Linux desktop integration

Canonical upstream assets for integrating GNOVI Studio into a Linux
desktop. They are **not** installed by `pip`; a distribution package
(Debian, Fedora, Arch, a Flatpak, …) is expected to place them in the
standard XDG locations.

## Files

| Asset | Source path | Install to |
|---|---|---|
| Launcher entry | `packaging/linux/gnovi-studio.desktop` | `$PREFIX/share/applications/gnovi-studio.desktop` |
| Scalable icon | `gnovi_plot/resources/icons/gnovi-studio.svg` | `$PREFIX/share/icons/hicolor/scalable/apps/gnovi-studio.svg` |
| Raster icons | `gnovi_plot/resources/icons/gnovi-studio-<N>.png` (N = 48, 128, 256, 512) | `$PREFIX/share/icons/hicolor/<N>x<N>/apps/gnovi-studio.png` |

The raster icons and the SVG are shipped inside the Python package
(`gnovi_plot/resources/icons/`), so they are present in both the wheel and
the sdist and are the single source of truth — the running application
loads the same PNGs for its window/taskbar icon.

## Notes for packagers

- `Exec=gnovi-studio` resolves to the `gui-scripts` console entry point
  declared in `pyproject.toml` (`gnovi_plot.app:main`). Ensure that script
  is on `PATH` (a normal `pip install` / distro package does this).
- `Icon=gnovi-studio` is an icon **name**, resolved through the icon theme;
  it needs the icons above installed under `hicolor`.
- The application calls `QApplication.setDesktopFileName("gnovi-studio")`,
  so the `.desktop` base name must stay `gnovi-studio` for the desktop
  environment to associate GNOVI's windows with this entry.
- GNOVI does not open files given as command-line arguments, so the entry
  declares no `MimeType`.
- After installing, refresh the caches:
  `update-desktop-database` and `gtk-update-icon-cache` (or the
  distribution's equivalent).

## Relationship to `assets/identity/`

`assets/identity/` holds the design masters and brand guidance and is not
shipped in the package. The runtime icons here are rendered from
`assets/identity/gnovi-studio-icon.svg`; regenerate them with, e.g.:

```sh
for n in 48 128 256 512; do
  inkscape assets/identity/gnovi-studio-icon.svg --export-type=png \
    --export-filename=gnovi_plot/resources/icons/gnovi-studio-$n.png \
    --export-width=$n --export-height=$n
done
cp assets/identity/gnovi-studio-icon.svg gnovi_plot/resources/icons/gnovi-studio.svg
```
