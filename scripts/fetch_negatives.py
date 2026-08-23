"""Collects hard negatives from Wikimedia Commons. prd.md section 11.4, section 5.2.5a.

The detector's false positives are its bottleneck, and half of section 11.4's negative classes can be
manufactured - low-bitrate blocking, resampling, grain - while the other half cannot. Real optical
defocus, LED video walls, mesh fabric and pixel art are *content*: they have to be found.

They matter because they are the shapes a mosaic detector confuses with block averaging. A LED wall
is a bright grid with dark gaps. Mesh fabric is a regular lattice over a scene. Pixel art is
block-constant on a grid **by intent**. Defocus is the smooth low-pass a mosaic also produces. A
detector that has never seen any of them has no reason not to fire on all of them - and measured, it
fires on 82% of clean cartoon frames against a requirement of 0.5%.

**Politeness is not optional here.** Wikimedia rate-limits, and this asks for hundreds of files: the
User-Agent identifies the tool, requests are spaced, and a 429 waits **for as long as the server
asks**. Its 429 carries ``Retry-After: 600``; an exponential guess topping out near three minutes
lands back inside the same penalty every time, which is how a run can retry itself to death and
collect thirteen files.

Everything downloaded is CC-licensed or public domain, and every file's licence and author are
recorded next to it. Under D-11 nothing is redistributed, but the record is what makes prd.md
section 2.4's reversal question answerable later.

Usage:

    .venv/Scripts/python.exe scripts/fetch_negatives.py
    .venv/Scripts/python.exe scripts/fetch_negatives.py --classes led mesh --per-class 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

# Anything printed here can carry a Commons filename, and Commons filenames are in every language
# there is. The console on this machine is cp949, so one accented character in a *value* kills the
# run - and the ASCII rule that guards printed *literals* cannot see it, because the literal is
# fine and the interpolated value is not. Replacing is the right trade for a progress line.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
NEGATIVES = REPO / "training" / "datasets" / "negatives"
MANIFEST = REPO / "training" / "datasets" / "negatives.manifest.json"

API = "https://commons.wikimedia.org/w/api.php"

#: Wikimedia asks that automated clients identify themselves and say what they are for.
USER_AGENT = "DeMosaicStudio/0.1 (personal research corpus; contact via repository)"

#: Default seconds between requests. Deliberately generous: this runs once and the API is a
#: shared resource. Wikimedia throttles hard, and the throttle applies to downloads too, so
#: raise it rather than retrying into a wall.
DEFAULT_DELAY = 2.5

#: The classes section 11.4 says cannot be manufactured, and where Commons keeps them.
CLASSES: dict[str, list[str]] = {
    # Real optical defocus. A mosaic is also a low-pass, so a detector with no defocus in its
    # negatives has no reason to tell them apart.
    "bokeh": ["Bokeh"],
    # A bright grid with dark gaps between emitters - structurally a mosaic with the contrast
    # inverted.
    "led": ["LED displays", "Video walls"],
    # A regular lattice laid over a scene.
    #
    # **Not `Tulle`.** That was the first choice and it is Commons' category for the *French commune
    # of Tulle*, not the fabric - the collected files were municipal photography and the measurement
    # taken from them was of French town squares. Checking what a category actually contains is not
    # optional, and a name that reads right in English is not a check.
    "mesh": ["Wire netting", "Mosquito nets"],
    # Block-constant on a grid by intent rather than by damage. The hardest class of the four,
    # because it is what a mosaic looks like when nothing is wrong.
    #
    # **Under-collected, and it cannot be fixed by synthesis.** Commons has almost no pixel art -
    # most of it is copyrighted - and `Category:Pixel art` is empty. Manufacturing it is worse than
    # having none: a photo downscaled with nearest-neighbour and scaled back up *is* a mosaic, so
    # labelling one as a negative teaches the detector the exact opposite of the thing. Real pixel
    # art differs by having a limited palette, deliberate outlines and high contrast between
    # adjacent blocks, and none of that survives a naive synthesis.
    "pixel-art": ["Video game screenshots"],
}

#: Below this on the shorter side an image is too small to crop a detector patch out of.
MIN_SIDE = 400

#: Above this an original is not worth the bandwidth for a negative patch source.
MAX_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Negative:
    """One collected file and where it came from."""

    name: str
    negative_class: str
    category: str
    title: str
    licence: str
    author: str
    source_url: str
    width: int
    height: int
    sha256: str
    size_bytes: int


def _retry_after(error: urllib.error.HTTPError, fallback: float) -> float:
    """How long the server said to wait.

    Wikimedia answers a 429 with ``Retry-After: 600``. Guessing instead - doubling from a couple of
    seconds - never waits long enough, so every attempt lands inside the same penalty and the run
    dies having learnt nothing. The header is the answer; read it.
    """
    header = error.headers.get("Retry-After") if error.headers else None
    if header:
        try:
            return max(float(header), fallback)
        except ValueError:
            pass
    return fallback


def _request(params: dict[str, str | int], delay: float) -> dict:
    """One API call, with the back-off a shared service is owed."""
    url = f"{API}?format=json&{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            wait = _retry_after(error, delay * (2 ** attempt))
            print(f"    throttled; the server asked for {wait:.0f}s", flush=True)
            time.sleep(wait)

    raise RuntimeError("still throttled after five attempts; try again later")


def _strip_html(text: str) -> str:
    """Commons returns small HTML fragments for author and licence. Keep the words."""
    out, depth = [], 0
    for character in text:
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif depth == 0:
            out.append(character)
    return " ".join("".join(out).split())


def files_in(category: str, limit: int, delay: float) -> list[dict]:
    """Lists files in a category with the metadata needed to record and fetch them."""
    result = _request({
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": limit,
        "prop": "imageinfo",
        # **No iiurlwidth.** Asking for a scaled rendering makes Commons generate it on request,
        # and that path is rate-limited far harder than serving an original from cache: at a 6 s
        # delay it still refused after a 192 s back-off. Originals come straight out.
        "iiprop": "url|size|extmetadata",
    }, delay)
    time.sleep(delay)

    pages = (result.get("query") or {}).get("pages") or {}
    out = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        if not info.get("thumburl") and not info.get("url"):
            continue

        out.append({
            "title": page.get("title", ""),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "url": info.get("url"),
            "bytes": info.get("size", 0),
            "descriptionurl": info.get("descriptionurl", ""),
            "licence": _strip_html(meta.get("LicenseShortName", {}).get("value", "unknown")),
            "author": _strip_html(meta.get("Artist", {}).get("value", "unknown"))[:120],
        })
    return out


def fetch(url: str, destination: Path, delay: float) -> bytes:
    """Downloads one file, backing off exactly as the API calls do.

    The first version backed off on the API and not on the downloads, which is where almost all the
    requests are. Every file after the second was refused, and the run reported "collected 2" as
    though that were a result.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()

            destination.write_bytes(payload)
            time.sleep(delay)
            return payload
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            wait = _retry_after(error, delay * (2 ** attempt))
            print(f"    throttled; the server asked for {wait:.0f}s", flush=True)
            time.sleep(wait)

    raise RuntimeError("still throttled after six attempts; try again later")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--classes", nargs="+", default=sorted(CLASSES),
                        choices=sorted(CLASSES), help="which negative classes to collect")
    parser.add_argument("--per-class", type=int, default=40,
                        help="how many files to keep per class")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help="seconds between requests; raise it if the API keeps throttling")
    args = parser.parse_args(argv)

    NEGATIVES.mkdir(parents=True, exist_ok=True)
    collected: list[Negative] = []

    existing: list[dict] = []
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8")).get("files", [])
    kept_from_before = [row for row in existing if row.get("negative_class") not in set(args.classes)]

    def record() -> None:
        """Writes the manifest after **every** file.

        The first version wrote it once, at the end. Wikimedia throttled the run to death partway
        through and 38 downloaded files were left with no recorded licence or author - which under
        this repository's own rule makes them unusable, so they had to be deleted. A collection that
        only records provenance if it finishes is a collection that loses it.
        """
        MANIFEST.write_text(
            json.dumps({
                "version": 1,
                "note": ("Hard negatives for the detector (prd.md section 11.4). Every file is "
                         "CC-licensed or public domain and its licence and author are recorded. "
                         "Under D-11 nothing is redistributed; this record answers prd.md "
                         "section 2.4 later."),
                "source": "Wikimedia Commons",
                "files": kept_from_before + [asdict(n) for n in collected],
            }, indent=2),
            encoding="utf-8",
        )

    for negative_class in args.classes:
        categories = CLASSES[negative_class]
        print(f"\n{negative_class}: {', '.join(categories)}", flush=True)

        # Re-collecting a class drops its manifest entries, so its files on disk have to go
        # with them. Leaving them behind produces exactly what this tool exists to avoid:
        # image files whose licence and author nothing records.
        for stale in NEGATIVES.glob(f"{negative_class}_*"):
            stale.unlink()

        kept = 0

        for category in categories:
            if kept >= args.per_class:
                break

            try:
                listing = files_in(category, min(100, args.per_class * 3), args.delay)
            except (urllib.error.URLError, RuntimeError) as exc:
                print(f"  {category}: {exc}", flush=True)
                continue

            for entry in listing:
                if kept >= args.per_class:
                    break
                if min(entry["width"], entry["height"]) < args.min_side:
                    continue
                if entry.get("bytes", 0) > MAX_BYTES:
                    continue
                # PDFs and video land in these categories too; the detector reads still frames.
                if not entry["url"].lower().split("?")[0].endswith((".jpg", ".jpeg", ".png")):
                    continue

                name = f"{negative_class}_{kept:03d}{Path(entry['url'].split('?')[0]).suffix.lower()}"
                destination = NEGATIVES / name
                try:
                    payload = fetch(entry["url"], destination, args.delay)
                except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                    print(f"  skipped {entry['title'][:40]}: {exc}", flush=True)
                    continue

                collected.append(Negative(
                    name=name,
                    negative_class=negative_class,
                    category=category,
                    title=entry["title"],
                    licence=entry["licence"],
                    author=entry["author"],
                    source_url=entry["descriptionurl"],
                    width=entry["width"],
                    height=entry["height"],
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                ))
                kept += 1
                record()
                print(f"  [{kept:3}] {name}  {entry['licence']}", flush=True)

        if kept == 0:
            print(f"  nothing usable in {', '.join(categories)}", flush=True)

    record()

    by_class: dict[str, int] = {}
    for negative in collected:
        by_class[negative.negative_class] = by_class.get(negative.negative_class, 0) + 1

    print()
    print(f"collected {len(collected)} files: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_class.items())))
    print(f"manifest: {MANIFEST.relative_to(REPO)}")

    return 0 if collected else 1


if __name__ == "__main__":
    sys.exit(main())
