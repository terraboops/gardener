"""CI gate: the gardener package must never import omlx (port, don't depend)."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "gardener"
PATTERN = re.compile(r"^\s*(import\s+omlx|from\s+omlx[\s.])", re.MULTILINE)

def main() -> int:
    offenders = []
    for path in PKG.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if PATTERN.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        print("FAIL: omlx import found in:", *offenders, sep="\n  ")
        return 1
    print("OK: no omlx imports in gardener/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
