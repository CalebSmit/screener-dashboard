"""WCAG 2.x contrast ratios for candidate design tokens on the dark surfaces."""


def lum(hexc: str) -> float:
    r, g, b = (int(hexc[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def f(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ratio(a: str, b: str) -> float:
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


SURFACES = {"page #0d0d0d": "#0d0d0d", "card #1a1a19": "#1a1a19", "raised #222221": "#222221"}
TOKENS = {
    "text-primary #ffffff": "#ffffff",
    "text-secondary #c3c2b7": "#c3c2b7",
    "text-muted #898781": "#898781",
    "accent #3987e5": "#3987e5",
    "accent-text #5598e7": "#5598e7",
    "accent-text #6da7ec": "#6da7ec",
    "pos #0ca30c": "#0ca30c",
    "pos-text #2fbf2f": "#2fbf2f",
    "neg #d03b3b": "#d03b3b",
    "neg-text #e05d5d": "#e05d5d",
    "neg-text #e66767": "#e66767",
    "amber #fab219": "#fab219",
}

for sname, s in SURFACES.items():
    print(f"\n-- on {sname} --")
    for tname, t in TOKENS.items():
        r = ratio(t, s)
        tag = "AA-small" if r >= 4.5 else ("AA-large/ui" if r >= 3.0 else "FAIL<3")
        print(f"  {tname:26s} {r:5.2f}:1  {tag}")
