"""How fast does the pipeline run on a real clip, and what does it report while running?

Drives the real worker over stdio - the same client the desktop application is - for a fixed number
of seconds, prints every progress message, then cancels. The number that matters is the last `fps`.

This exists because the answer was wrong by an order of magnitude from what anyone assumed: 0.44
frames a second at 1080p with the shipped settings (D-42), so an hour of video was a 67-hour job and
nothing in the product said so. Every speed decision since has been measured with this.

Usage::

    .venv/Scripts/python.exe scripts/bench_throughput.py CLIP [--seconds 40] [--mask 0.5]
        [--min-area 1024] [--preset Balanced] [--detect-every 1]

Prints ASCII only: the console this runs in is cp949.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pipeline throughput on one clip")
    parser.add_argument("clip", type=Path)
    parser.add_argument("--seconds", type=float, default=40.0, help="run this long, then cancel")
    parser.add_argument("--mask", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=1024)
    parser.add_argument("--preset", default="Fast")
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--refine", default=None, help="diffusion model name under models/diffusion (D-44)")
    parser.add_argument("--lora", default=None, help="LoRA name under models/lora")
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = parser.parse_args(argv)

    import av

    with av.open(str(args.clip)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration or 0) / 1e6
        rate = float(stream.average_rate or 24)
        print("source   : %dx%d  %.1f s  %.2f fps  -> about %d frames"
              % (stream.width, stream.height, duration, rate, duration * rate))
    print("settings : preset=%s mask=%.2f minArea=%d detectEvery=%d refine=%s"
          % (args.preset, args.mask, args.min_area, args.detect_every, args.refine or "off"))

    output = Path(REPO / "artifacts" / "bench" / "throughput.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        [sys.executable, "-m", "demosaic_worker.main_loop"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "worker"), "SYSTEMROOT": r"C:\Windows",
             "PATH": r"C:\Windows\System32"},
    )
    assert process.stdin and process.stdout

    def send(**message: object) -> None:
        process.stdin.write(json.dumps({"v": "1.3", "id": "bench", **message}) + "\n")
        process.stdin.flush()

    send(type="hello", hostVersion="1.0", protocolVersion="1.3")
    send(type="process", jobId="bench", sourcePath=str(args.clip), outputPath=str(output),
         settings={
             "detection": {"confidence": 0.45, "maskThreshold": args.mask,
                           "minRegionArea": args.min_area, "detectEvery": args.detect_every},
             "restoration": {"preset": args.preset, "temporalWindow": "auto",
                             "refine": {"enabled": bool(args.refine), "model": args.refine or "",
                                        "lora": args.lora, "strength": args.strength, "steps": 8, "seed": 7}},
             "encode": {"codec": "H265", "constantQuality": 18, "preset": "medium"},
         })

    started = time.monotonic()
    cancelled = False
    last_fps = None
    frames_reported = 0

    for line in process.stdout:
        try:
            message = json.loads(line)
        except ValueError:
            continue

        elapsed = time.monotonic() - started
        if message["type"] == "progress":
            if message.get("fps"):
                last_fps = message["fps"]
                frames_reported += 1
            if not args.quiet:
                print("  %6.1fs  %-11s %5.1f%%  fps=%s  eta=%s"
                      % (elapsed, message["stage"], 100 * message["fraction"],
                         message.get("fps"), message.get("eta")), flush=True)
        elif message["type"] == "log" and message.get("level") == "warning" and not args.quiet:
            print("  warn: %s" % message.get("message"), flush=True)
        elif message["type"] == "log" and "refiner" in str(message.get("message", "")) and not args.quiet:
            print("  log: %s" % message.get("message"), flush=True)
        elif message["type"] == "result":
            print("RESULT %s after %.1fs   last fps=%s   regionsRefined=%s"
                  % (message["status"], elapsed, last_fps, (message.get("summary") or {}).get("regionsRefined")))
            send(type="shutdown")
            process.wait(timeout=30)
            output.unlink(missing_ok=True)
            return 0

        if elapsed > args.seconds and not cancelled:
            cancelled = True
            send(type="cancel", jobId="bench")

    return 1


if __name__ == "__main__":
    sys.exit(main())
