"""Command-line front end for the worker. prd.md section 8.

There is no desktop application yet (Phase 4). This is the smallest thing that makes the pipeline
usable: it speaks the same protocol the WPF host will speak, over the same stdio transport, so
anything that works here works there.

It is deliberately a *client*, not a shortcut - it launches `demosaic_worker.main_loop` as a child
process and talks JSON Lines to it. A version that imported the runner directly would be easier to
write and would stop testing the boundary that matters.

NOTE: this docstring is argparse's --help text, so it is ASCII only. A section mark here crashes
--help on a cp949 console, which is the third time that trap has been sprung in this repository.
See AGENTS.md and CLAUDE.md section 4.

Usage:

    .venv/Scripts/python.exe scripts/run_job.py INPUT.mp4 OUTPUT.mp4
    .venv/Scripts/python.exe scripts/run_job.py INPUT.mp4 --probe-only
    .venv/Scripts/python.exe scripts/run_job.py INPUT.mp4 OUT.mp4 --threshold 0.9 --crf 12

What this will do to your video, honestly: on the one clip it has been measured against, the output
scores about 0.7 dB *below* the input inside the mosaicked region and about 4.6 dB below it
elsewhere (docs/phase3-endtoend-report.md section 8.3). It runs; it does not yet improve anything.
Use --dry-run to see what it would detect without writing a file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"


def _requests(args: argparse.Namespace) -> list[dict]:
    messages: list[dict] = [{"v": "1.0", "type": "hello", "id": "1", "hostVersion": "cli"}]

    messages.append({"v": "1.0", "type": "probe", "id": "2", "sourcePath": str(args.source)})

    if not args.probe_only:
        settings = {
            "detection": {
                "confidence": args.confidence,
                "maskThreshold": args.threshold,
                "minRegionArea": args.min_area,
                "minConfirmFrames": args.confirm_frames,
            },
            "restoration": {
                "preset": args.preset,
                "temporalWindow": args.window,
                "minRestorationConfidence": args.min_confidence,
            },
            "encode": {
                "profile": "QualityX265",
                "codec": args.codec,
                "constantQuality": args.crf,
                "preset": args.encoder_preset,
            },
        }
        if args.model_version:
            settings["modelVersion"] = args.model_version

        messages.append(
            {
                "v": "1.0", "type": "analyze" if args.dry_run else "process", "id": "3",
                "jobId": "cli-1",
                "sourcePath": str(args.source),
                "outputPath": str(args.output) if args.output else "",
                "settings": settings,
                "resume": False,
            }
        )

    messages.append({"v": "1.0", "type": "shutdown", "id": "4"})
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--probe-only", action="store_true", help="report media facts and stop")
    parser.add_argument("--dry-run", action="store_true", help="detect and report, write nothing")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="mask binarization threshold; 0.9 is the calibrated point (default)")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=1024)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--preset", default="Balanced", choices=["Fast", "Balanced", "Quality"])
    parser.add_argument("--window", default="auto", help="auto, 3, 5, 7 or 9")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="withhold restorations below this confidence (0 = off)")
    parser.add_argument("--crf", type=int, default=12,
                        help="12 is the measured transparent point for x265 (default)")
    parser.add_argument("--codec", default="H265", choices=["H264", "H265"])
    parser.add_argument("--encoder-preset", default="fast")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--verbose", action="store_true", help="print every protocol message")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"source not found: {args.source}", file=sys.stderr)
        return 2
    if not args.probe_only and not args.dry_run and args.output is None:
        parser.error("an output path is required unless --probe-only or --dry-run")

    if not PYTHON.exists():
        print(f"worker interpreter missing: {PYTHON}\nRun scripts/setup-worker.ps1", file=sys.stderr)
        return 2

    started = time.time()
    proc = subprocess.Popen(
        [str(PYTHON), "-m", "demosaic_worker.main_loop"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "worker"), "SYSTEMROOT": r"C:\Windows",
             "PATH": r"C:\Windows\System32"},
    )

    stdout, stderr = proc.communicate(
        "\n".join(json.dumps(r) for r in _requests(args)) + "\n"
    )

    status = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue

        message = json.loads(line)
        kind = message["type"]

        if args.verbose:
            print(json.dumps(message)[:200])
            continue

        if kind == "ready":
            caps = message["capabilities"]
            device = caps.get("device", "cpu")
            models = ", ".join(f"{m['id']} {m['version']}" for m in caps.get("models", []))
            print(f"worker {message['workerVersion']}, protocol {message['protocolVersion']}")
            print(f"  device: {device if caps.get('cudaAvailable') else 'CPU only'}")
            print(f"  models: {models or 'none'}")

        elif kind == "probeResult":
            m = message["media"]
            print(f"\n{args.source.name}")
            print(f"  {m['width']}x{m['height']}  {m['videoCodec']}  "
                  f"{'VFR' if m['isVfr'] else 'CFR'}  {m['durationSeconds']:.2f}s")
            print(f"  audio: {len(m['audioStreams'])}  subtitles: {len(m['subtitleStreams'])}")

        elif kind == "progress":
            fraction = message.get("fraction") or 0.0
            fps = message.get("fps")
            bar = "#" * int(fraction * 30)
            sys.stdout.write(
                f"\r  {message['stage']:<10} [{bar:<30}] {fraction:5.1%}"
                + (f"  {fps:.1f} fps" if fps else "")
            )
            sys.stdout.flush()

        elif kind == "log":
            if message["level"] in ("warning", "error"):
                print(f"\n  ! {message.get('code') or ''} {message['message'][:120]}")

        elif kind == "error":
            print(f"\n  ERROR {message['code']}: {message['message']}", file=sys.stderr)
            status = 1

        elif kind == "result":
            s = message["summary"]
            print(f"\n\nstatus: {message['status']}   ({time.time() - started:.0f}s)")
            print(f"  frames        {s['framesSeen']}  "
                  f"({s['framesRestored']} restored, {s['framesPassedThrough']} passed through)")
            print(f"  regions       {s['regionsDetected']}"
                  + (f"  ({s['regionsGated']} withheld)" if s.get("regionsGated") else ""))
            print(f"  timeline      {s.get('timeline', 'n/a')}")
            print(f"  confidence    {s['confidenceMean']:.3f} mean")
            if s.get("routeReasons"):
                print("  routing:")
                for reason, count in sorted(s["routeReasons"].items(), key=lambda kv: -kv[1]):
                    print(f"    {count:6}  {reason}")
            if s.get("synthetic"):
                print("\n  Output contains synthetic content: where information was destroyed the")
                print("  restoration is an estimate, not recovered original pixels (prd.md 1.3).")
            if message["status"] == "failed":
                status = 1

    if proc.returncode != 0 and status == 0:
        print(f"\nworker exited {proc.returncode}", file=sys.stderr)
        print(stderr.strip()[-1500:], file=sys.stderr)
        status = 1

    return status


if __name__ == "__main__":
    sys.exit(main())
