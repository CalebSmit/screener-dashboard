"""Render the site's home-screen icons from the mark every page already uses as its favicon.

WHY (round three, 2026-10-27, plan/ui-elevation.md): the pages carry an SVG favicon only. iOS ignores
SVG for "Add to Home Screen" and shows a screenshot of the page instead, and a web app manifest needs
PNG icons at 192 and 512 px. Students open this on a phone; the icon is how they find it again.

    python scripts/make_icons.py

Writes, at the repository root where GitHub Pages serves them:

* ``icon-180.png`` - ``apple-touch-icon`` (iOS rounds the corners itself, so the square is full-bleed)
* ``icon-192.png``, ``icon-512.png`` - the manifest's ``any`` icons, the favicon's rounded square
* ``icon-maskable-512.png`` - the manifest's ``maskable`` icon: full-bleed ground, the bars inside the
  central 80% safe zone so a launcher's circle or squircle never cuts them

The geometry is the favicon's own 32-unit drawing (``generate_dashboard.generate_html``'s
``rel="icon"``): ground #141413, three bars in the accent #3987e5 at 55%, 80% and 100% opacity. Drawn at
4x and downsampled, so the edges are smooth. Needs Pillow (already a dependency of the test run).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
GROUND = (0x14, 0x14, 0x13)
ACCENT = (0x39, 0x87, 0xE5)
#: (x, y, w, h, opacity) in the favicon's 32-unit box
BARS = ((7, 17, 4, 8, 0.55), (14, 12, 4, 13, 0.80), (21, 7, 4, 18, 1.0))
SS = 4  # supersampling


def _mix(alpha: float) -> tuple[int, int, int]:
    return tuple(round(g + (a - g) * alpha) for g, a in zip(GROUND, ACCENT))


def render(size: int, *, rounded: bool, inset: float = 0.0) -> Image.Image:
    """``inset`` shrinks the bars toward the centre (0.0 = the favicon's own placement)."""
    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if rounded:
        d.rounded_rectangle((0, 0, big - 1, big - 1), radius=big * 8 / 32, fill=GROUND + (255,))
    else:
        d.rectangle((0, 0, big - 1, big - 1), fill=GROUND + (255,))
    scale = big / 32 * (1 - inset)
    off = big * inset / 2
    for x, y, w, h, a in BARS:
        box = (off + x * scale, off + y * scale, off + (x + w) * scale, off + (y + h) * scale)
        d.rounded_rectangle(box, radius=1.5 * scale, fill=_mix(a) + (255,))
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    outputs = {
        "icon-180.png": render(180, rounded=False),
        "icon-192.png": render(192, rounded=True),
        "icon-512.png": render(512, rounded=True),
        # the bars span x 7-25 / y 7-25 of 32: their farthest corner is 12.7 units from the centre, the
        # safe circle's radius 12.8 - so scaled to 80% (10.2 units) they sit well inside it
        "icon-maskable-512.png": render(512, rounded=False, inset=0.2),
    }
    for name, img in outputs.items():
        img.save(ROOT / name, optimize=True)
        print(f"wrote {name} ({(ROOT / name).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
