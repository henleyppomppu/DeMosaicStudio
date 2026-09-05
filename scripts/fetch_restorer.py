"""Installs the compact super-resolution weights the Balanced preset uses. D-43.

Downloads ``realesr-general-x4v3.pth`` from the Real-ESRGAN GitHub release, records its SHA-256 and
provenance, and writes it into the model store in the store's own layout::

    models/restorer/realesr-general-x4v3/
        model.pt          {"state_dict": ..., "num_feat": 64, "num_conv": 32}
        metadata.json     id, version, sha256 of model.pt, source URL, licence

The weights are not in the repository (``models/`` is ignored, like the detector's). Without them
the Balanced preset falls back to Fast with warning W6101, so nothing breaks - it just does not
invent detail.

**What this is and is not.** A general-purpose photographic super-resolution network, BSD-3. It is
not a face model: prd.md section 2.3 C-2 and C-4 rule those out permanently, and CodeFormer/GFPGAN
must never be substituted here however much better a face would look.

Usage::

    .venv/Scripts/python.exe scripts/fetch_restorer.py [--force]

Prints ASCII only: the console this runs in is cp949.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))

from demosaic_worker.restore.upscale import RESTORER_ID  # noqa: E402

SOURCE_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
)
LICENCE = "BSD-3-Clause (Real-ESRGAN, Xintao Wang et al.)"
DESTINATION = REPO / "models" / "restorer" / RESTORER_ID


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="install the Balanced preset's restorer weights")
    parser.add_argument("--force", action="store_true", help="re-download even if installed")
    args = parser.parse_args(argv)

    import torch

    weights = DESTINATION / "model.pt"
    metadata = DESTINATION / "metadata.json"
    if weights.exists() and metadata.exists() and not args.force:
        print("already installed:", DESTINATION)
        return 0

    DESTINATION.mkdir(parents=True, exist_ok=True)
    downloaded = DESTINATION / "realesr-general-x4v3.pth"

    print("downloading", SOURCE_URL)
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, downloaded.open("wb") as out:
        total = 0
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
            total += len(chunk)
    print("received %.1f MB, sha256 %s" % (total / 2**20, _sha256(downloaded)))

    # Re-save in the store's layout so the loader's hash covers exactly what it loads.
    checkpoint = torch.load(downloaded, map_location="cpu", weights_only=True)
    state = next(
        (checkpoint[key] for key in ("params_ema", "params", "state_dict") if key in checkpoint),
        checkpoint,
    )
    torch.save({"state_dict": state, "num_feat": 64, "num_conv": 32}, weights)
    downloaded.unlink()

    metadata.write_text(json.dumps({
        "id": RESTORER_ID,
        "version": "0.2.5.0",
        "task": "single-image super-resolution x4 (decimated mosaic crops, luma)",
        "sha256": _sha256(weights),
        "architecture": {"type": "SRVGGNetCompact", "numFeat": 64, "numConv": 32, "upscale": 4},
        "source": SOURCE_URL,
        "license": LICENCE,
        "notes": (
            "Third-party weights used as-is on decimated input (D-43, revising D-04). "
            "General-purpose photographic prior; not a face model (prd.md 2.3 C-2, C-4)."
        ),
    }, indent=2) + "\n", encoding="utf-8")

    print("installed:", DESTINATION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
