# Troubleshooting

Symptom → cause → fix, keyed to the error codes in `docs/ERROR_CODES.md`.

**Status: seeded from the codes that exist and from failures actually hit during development.** Paths
that are not implemented yet say so rather than offering invented remedies.

## Build and environment

| Symptom | Cause | Fix |
| --- | --- | --- |
| `MSBUILD : error MSB1009: project file does not exist. Switch: DeMosaicStudio.sln` | .NET 10 generates `.slnx`, not `.sln` | `dotnet build DeMosaicStudio.slnx` |
| `python` opens the Microsoft Store | The Store alias shadows a real interpreter | Use `.\.venv\Scripts\python.exe`, always |
| `ModuleNotFoundError: No module named 'demosaic_worker'` | `pytest.ini` sets `pythonpath`; a bare invocation does not | Run pytest from the repo root, or set `PYTHONPATH=worker` |
| `error CA1720: identifier contains type name` | Warnings are errors and an enum member collides with a type name | Rename it — `ObjectTracked`, not `Object` |
| Test method names fail `CA1707` | Naming analyzers applied to test projects | Suppressed in `tests/Directory.Build.props`. If it reappears, check that file still imports the root props via `GetPathOfFileAbove` — without that import the entire root configuration is silently ignored |
| A `.ps1` fails to parse with a nonsense syntax error, on the target machine only | Saved without a UTF-8 BOM; PowerShell 5.1 reads it as CP949 and multi-byte characters break tokenisation | Re-save as **UTF-8 with BOM**. `ScriptEncodingTests` catches this |
| `§` renders as garbage in script output | The file is fine; the *console* is CP949 | Keep `Write-Host` strings ASCII. Comments and docstrings may use anything |
| `SyntaxWarning: invalid escape sequence` from a Python script | A Windows path in a normal string | Use a raw string: `r"C:\..."` |
| `UnicodeEncodeError` when logging Korean or Japanese | `sys.stdout.encoding` is cp949 | Call `demosaic_worker.stdio.configure_stdio_utf8()` before writing anything |
| `ModuleNotFoundError: No module named 'tests.test_...'` | Two directories both named `tests` with `__init__.py` become the same package | Do not put `__init__.py` in test directories |

## Media (E1xxx, E2xxx)

| Code | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| E1001 | Job fails immediately | Source missing or unreadable | Check the path. Failing immediately rather than after minutes of decoding is deliberate |
| E1002 / E1003 | Rejected at probe | Container or codec the installed FFmpeg cannot decode | Remux or transcode the source first |
| E1004 / E2003 | "Invalid data found when processing input" | Truncated or corrupt file | Verify it plays elsewhere. A truncated MP4 fails at *open*, not mid-stream |
| E1006 | Warning, job continues | Declared duration disagrees with what decoded | Usually harmless — the pipeline is PTS-driven, not duration-driven |
| E2001 → W1101 | Slower than expected, `decoder="software"` in the result | Hardware decoder init failed | Not an error. Investigate the driver only if it happens every time |
| E2004 | Timestamp discontinuity | Concatenated or spliced source | Usually recoverable; compare output frame count against the source |

## Audio and timing

| Symptom | Cause | Fix |
| --- | --- | --- |
| Output audio differs from the source | Audio was transcoded | It must be stream-copied (§5.1.5). `T-IO-AUDIO-COPY-01` guards this |
| Audio stream hashes differ but the audio sounds identical | The hash was taken after `container.seek(0)`, which drops the codec pre-roll (AAC priming) | Open the container fresh per stream; do not seek |
| Subtitles drift late in a long file | Output written at a synthesized constant rate instead of the source PTS | §5.1.7 — output frame *count* and per-frame PTS must both match |
| Output is VFR when the source was CFR, or the reverse | Same cause | `T-IO-PTS-CFR-01`, `T-IO-PTS-VFR-01` |

## Detection and restoration (E3xxx, E4xxx)

| Code | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| E3001 / E4001 | Job fails before processing | Model missing or hash mismatch | Check `models/index.json`. §14.3 requires this to fail *before* decoding starts |
| E3003 | Shape mismatch | Model and runtime disagree | The model's `metadata.json` input requirements are authoritative |
| E4004 | Region skipped | ROI below the model minimum | Expected; the region passes through unmodified |
| E4401 | Job fails after the OOM ladder | VRAM exhausted at every step | Lower the VRAM budget or the preset. Output is discarded rather than left partially composited |
| W4102 | "N regions left untouched" in the summary | `minRestorationConfidence` withheld them (§5.8.1) | Working as configured. Lower the threshold to restore them, accepting weaker evidence |
| W4103 | Effective K smaller than requested | A safety rule reduced it: scene cut, object-anchored grid, VRAM, or stream boundary | Expected. The reason is carried in the warning |
| W5102 | "could not stream-copy" on a job that restored nothing | No ffmpeg on this machine - `tools/ffmpeg` is gitignored and is part of the install, not the checkout | Install `tools/ffmpeg`, or set `DEMOSAIC_FFMPEG`. The output is usable either way, just re-encoded and about 2.9 dB softer (D-22) |

**Restoration produced output that looks worse than the source.** Check the diagnostic overlay for
grid anchoring and motion band. Object-anchored multi-frame is measurably harmful
(`docs/phase0-report.md` §3.2) and so is multi-frame on fast-motion content
(`docs/phase2-alignment-report.md` §3). Both should have been gated by §5.8; if they were not, that
is a router bug, not a tuning problem.

**The output is a whole new file even though nothing was detected.** Check `passthrough` in the
summary. When it is `true` the video bitstream is byte-identical to the source and only the
container was rewritten, so the file size and hash of the *video stream* match; the container bytes
legitimately differ. When it is `false` on a job that restored nothing, look for W5102 above.

**A detector that fires on clean footage costs more than a few wrong pixels.** It also takes the
pass-through away: on one clean corpus clip the detector found 48 regions across 96 frames, so the
file was fully re-encoded for nothing. Pass-through is worth exactly as much as the false-positive
rate allows (`docs/untouched-decomposition.json`).

## Resume and checkpoints (§9)

| Symptom | Cause | Fix |
| --- | --- | --- |
| A retry redoes work that had finished | A fingerprint changed — check which artifact was discarded | Expected if a setting in that scope changed. **Not** expected for VRAM budget, precision, tile size, batch size or comparison points; those are excluded (§9.3) |
| Every retry restarts from zero | Fingerprints are not rewritten immediately after a discard | §9.3 — the rewrite happens at discard time, not at job completion |
| Resume reuses artifacts from a different source file | A null fingerprint compared as "equal" | §9.3 — unknown must compare as **changed**. `T-RESUME-FINGERPRINT-05` |

## Protocol (E7xxx)

| Code | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| E7001 | Worker refused at handshake | Major protocol version differs | Host and worker are from different builds |
| E7002 | No `ready` within 10 s | Worker failed to start | Check `worker.log` in the job directory, and the venv |
| E7005 | Job fails, application keeps running | Worker crashed | By design: one job dies, the host relaunches for the next |
| E7006 | Malformed message | A `print()` went to stdout instead of stderr | stdout is the protocol channel; all human-readable output goes to stderr |
