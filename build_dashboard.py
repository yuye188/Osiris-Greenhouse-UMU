"""
build_dashboard.py
==================
Assemble the standalone HTML audit dashboard.

The page is one self-contained file: no server, no network calls, no data
fetching at run time. All the numbers are inlined as a single JSON blob, so the
result can be opened by double-clicking it, e-mailed, or committed.

Pipeline
--------
    python export_dashboard_data.py     # runs every check -> protocol check output/dash.json
    python build_dashboard.py           # dash.json + template -> osiris_campaign_audit.html

``dashboard_template.html`` is the page with a ``__DATA__`` placeholder where the
JSON goes. Edit the template for anything about how the page looks or behaves;
never edit the built file, because the next build overwrites it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TEMPLATE = Path("dashboard_template.html")
DATA = Path("protocol check output") / "dash.json"
OUTPUT = Path("osiris_campaign_audit.html")
PLACEHOLDER = "__DATA__"


def main() -> int:
    for f in (TEMPLATE, DATA):
        if not f.exists():
            print(f"missing {f}", file=sys.stderr)
            if f is DATA:
                print("run export_dashboard_data.py first", file=sys.stderr)
            return 1

    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        print(f"{TEMPLATE} has no {PLACEHOLDER} placeholder", file=sys.stderr)
        return 1

    raw = DATA.read_text(encoding="utf-8")
    json.loads(raw)                       # fail loudly on malformed data
    # The blob lives inside a <script> element, so any literal "</" in it would
    # close that element early. Escaping the slash keeps the JSON valid.
    out = html.replace(PLACEHOLDER, raw.replace("</", r"<\/"))
    OUTPUT.write_text(out, encoding="utf-8")

    print(f"wrote {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    print("open it by double-clicking, or drag it into a browser window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
