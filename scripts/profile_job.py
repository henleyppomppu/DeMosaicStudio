"""Where does a frame's time actually go? Runs one job in-process under cProfile.

The throughput bench (bench_throughput.py) says how fast; this says why. It exists because the
first speed-first rewrite removed the part everyone assumed was the cost - per-region optical flow
and the solver - and the frame rate moved from 0.74 to 1.0. The assumption was wrong, and the only
way to find out what was actually slow was to measure every function.

Usage::

    .venv/Scripts/python.exe scripts/profile_job.py CLIP [--seconds 20] [--preset Fast] [--top 25]

Prints ASCII only: the console this runs in is cp949.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))

from demosaic_worker.jobs import JobContext, JobRunner  # noqa: E402
from demosaic_worker.messages import Emitter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="profile one job in-process")
    parser.add_argument("clip", type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--preset", default="Fast")
    parser.add_argument("--mask", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=1024)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--sort", default="cumulative", choices=["cumulative", "tottime"])
    args = parser.parse_args(argv)

    output = REPO / "artifacts" / "bench" / "profile.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    context = JobContext(
        job_id="profile",
        source_path=str(args.clip),
        output_path=str(output),
        settings={
            "detection": {"confidence": 0.45, "maskThreshold": args.mask,
                          "minRegionArea": args.min_area, "detectEvery": args.detect_every},
            "restoration": {"preset": args.preset, "temporalWindow": "auto"},
            "encode": {"codec": "H265", "constantQuality": 18, "preset": "medium"},
        },
    )

    # Progress goes nowhere; the frame count comes from the summary.
    emitter = Emitter(stream=io.StringIO())

    def stop_later() -> None:
        time.sleep(args.seconds)
        context.cancelled = True

    threading.Thread(target=stop_later, daemon=True).start()

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    summary = JobRunner().run(context, emitter)
    profiler.disable()
    elapsed = time.perf_counter() - started

    frames = int(summary.get("framesSeen", 0))
    print("preset=%s  frames=%d  elapsed=%.1fs  ->  %.1f ms/frame  (%.2f fps)"
          % (args.preset, frames, elapsed, 1000 * elapsed / max(frames, 1), frames / elapsed))
    print()

    stats = pstats.Stats(profiler)
    stats.sort_stats(args.sort)
    buffer = io.StringIO()
    stats.stream = buffer
    stats.print_stats(args.top)
    # Trim the repository path out of every line so the table fits.
    text = buffer.getvalue().replace(str(REPO) + "\\", "").replace(str(REPO) + "/", "")
    sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))

    output.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
