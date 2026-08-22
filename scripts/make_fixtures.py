"""Generates the tiny media fixtures the tests run against. prd.md §13.3.

Small enough to commit, varied enough to catch the timing bugs that matter: CFR, VFR, rotation
metadata, multiple audio tracks, and a deliberately truncated file.

Run with the worker venv:

    .venv/Scripts/python.exe scripts/make_fixtures.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
OUT = REPO / "fixtures" / "media"


def run(args: list[str]) -> None:
    result = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {' '.join(args)}\n{result.stderr}")


def main() -> int:
    if not FFMPEG.exists():
        raise SystemExit(f"FFmpeg not found at {FFMPEG}. Run scripts/setup-worker.ps1 first.")

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. CFR, 30 fps, 20 frames, with one audio track.
    run([
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=0.6667",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.6667",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "32k",
        str(OUT / "cfr_30fps.mp4"),
    ])

    # 2. VFR: the same content with an uneven timestamp pattern. Built by setting a variable
    #    frame rate through the fps filter's expression support and then muxing with -vsync passthrough.
    run([
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=1",
        "-vf", "select='not(mod(n,3))+not(mod(n,7))'", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        str(OUT / "vfr.mp4"),
    ])

    # 3. Rotation metadata, which must survive a round trip (§5.1.6).
    run([
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=0.5",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-metadata:s:v:0", "rotate=90",
        str(OUT / "rotated_90.mp4"),
    ])

    # 4. Three audio tracks, to prove all of them are preserved bit-identically (§5.1.5).
    run([
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=0.5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
        "-f", "lavfi", "-i", "sine=frequency=660:duration=0.5",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=0.5",
        "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "32k",
        str(OUT / "multi_audio.mkv"),
    ])

    # 5. A truncated file: the decoder must fail with E2003/E1004 rather than produce silence.
    source = (OUT / "cfr_30fps.mp4").read_bytes()
    (OUT / "corrupt_truncated.mp4").write_bytes(source[: len(source) // 3])

    for path in sorted(OUT.iterdir()):
        print(f"{path.name:28} {path.stat().st_size:>8} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
