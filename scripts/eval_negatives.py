"""Does the detector fire on the things section 11.4 said it would? prd.md section 5.2.5a, section 11.4.

section 11.4 names four classes of content a mosaic detector is expected to confuse with block averaging:
real optical defocus, LED video walls, mesh fabric, and pixel art. That is a hypothesis, and the
corpus never contained any of them, so it has never been tested.

`scripts/fetch_negatives.py` collects them from Wikimedia Commons. This measures what the detector
does with them, next to clean film frames as the control - because a detector that fires on
everything is not evidence that these classes are special.

Usage:

    .venv/Scripts/python.exe scripts/eval_negatives.py
    .venv/Scripts/python.exe scripts/eval_negatives.py --model det-unet-0.1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))

from demosaic_worker.detect.regions import extract_regions  # noqa: E402
from demosaic_worker.detect.segmenter import Segmenter  # noqa: E402

NEGATIVES = REPO / "training" / "datasets" / "negatives"
MANIFEST = REPO / "training" / "datasets" / "negatives.manifest.json"
CORPUS = REPO / "training" / "datasets" / "clean"

#: Detector patches are cut from frames, so a huge photograph is downscaled to something
#: frame-shaped first. Scoring a 5000 px original would measure a resolution these images do not
#: share with video.
FRAME_HEIGHT = 800


def _as_frame(path: Path) -> np.ndarray | None:
    """Loads an image as a luma frame of roughly video height."""
    try:
        with Image.open(path) as image:
            grey = image.convert("L")
            scale = FRAME_HEIGHT / grey.height
            if scale < 1.0:
                grey = grey.resize(
                    (max(int(grey.width * scale), 16), FRAME_HEIGHT), Image.LANCZOS
                )
            return np.asarray(grey, dtype=np.float64)
    except (OSError, ValueError):
        return None


def _clean_frames(limit: int) -> list[np.ndarray]:
    """The control: frames from clean film, which the detector should also leave alone."""
    import av

    out: list[np.ndarray] = []
    for clip in sorted(CORPUS.glob("tos_*.mp4"))[:limit]:
        with av.open(str(clip)) as container:
            for frame in container.decode(container.streams.video[0]):
                out.append(frame.to_ndarray(format="gray").astype(np.float64))
                break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="det-unet-0.2.0")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.5, 0.9, 0.99, 0.999])
    parser.add_argument("--min-area", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "negatives-report.json")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        print(f"no negatives collected yet: run scripts/fetch_negatives.py", file=sys.stderr)
        return 2

    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    by_class: dict[str, list[np.ndarray]] = defaultdict(list)

    for record in records:
        frame = _as_frame(NEGATIVES / record["name"])
        if frame is not None:
            by_class[record["negative_class"]].append(frame)

    by_class["clean film (control)"] = _clean_frames(limit=24)

    segmenter = Segmenter(REPO / "models" / "detector" / args.model)
    print(f"{args.model} on {sum(len(v) for v in by_class.values())} images")
    print()
    print(f"{'class':24} {'n':>4} " + " ".join(f"{t:>8}" for t in args.thresholds))

    report: dict[str, dict[str, float]] = {}
    for name, frames in sorted(by_class.items()):
        if not frames:
            continue

        probabilities = [segmenter.probability(frame) for frame in frames]
        row = []
        for threshold in args.thresholds:
            firing = sum(
                1 for p in probabilities
                if extract_regions(p, threshold=threshold, min_area=args.min_area,
                                   max_regions=64)[0]
            )
            row.append(firing / len(probabilities))

        report[name] = {str(t): round(v, 4) for t, v in zip(args.thresholds, row)}
        print(f"{name:24} {len(frames):4} " + " ".join(f"{v:8.1%}" for v in row))

    args.out.write_text(json.dumps({
        "model": args.model,
        "minArea": args.min_area,
        "requirement": {"source": "prd.md section 5.2.5a", "maxFiringFraction": 0.005},
        "firing": report,
    }, indent=2), encoding="utf-8")

    print()
    print("section 5.2.5a asks for <= 0.5% of frames with no mosaic to produce any region.")
    print(f"wrote {args.out.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
