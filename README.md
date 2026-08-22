# DeMosaic Studio

Automatic detection, tracking and multi-frame restoration of partially mosaicked regions in video.
Windows desktop, local NVIDIA GPU, no cloud inference.

**This software is for personal use and is not distributed** (`prd.md` §2.4, D-11).

Use it only on video you own or are otherwise authorized to process. Reconstructed content is
**synthetic**: where information has been destroyed, the model estimates plausible content, and no
part of the output should be treated as recovered original pixels or as evidence of anything
(`prd.md` §1.3, §2.3).

## Status

**Phase 0 complete. Phases 1–6 are not.**

| Phase | State |
| --- | --- |
| 0 — foundations and the feasibility gate | **Complete.** Verdict `PASS_ALIGNMENT_BLOCKED` |
| 1 — detection and tracking | One experiment: a detector reaches 0.82 held-out IoU. The phase's exit criteria are not met |
| 2 — restoration | One experiment: dense alignment measured. No restoration model exists |
| 3–6 — pipeline, desktop app, optimisation, hardening | Not started |

Concretely: `DeMosaicStudio.Domain` is built and tested; `Application`, `Infrastructure` and `App`
are empty directories. The worker has media I/O and the protocol types but no dispatch loop and none
of the pipeline stages. **No video has ever been restored end to end, and no real mosaicked footage
has been processed at all.**

The measurements that shaped the plan are in
[`docs/phase0-report.md`](docs/phase0-report.md),
[`docs/phase1-detector-report.md`](docs/phase1-detector-report.md) and
[`docs/phase2-alignment-report.md`](docs/phase2-alignment-report.md). Each has a limitations section;
read it before quoting any number.

## Documents

| File | Purpose |
| --- | --- |
| [`prd.md`](prd.md) | Requirements. Single source of truth |
| [`prompt.md`](prompt.md) | Execution plan and phase breakdown |
| [`CLAUDE.md`](CLAUDE.md) | Handover notes: what is verified, what broke, what is next |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | ADRs. Check before reverting anything |
| [`docs/ERROR_CODES.md`](docs/ERROR_CODES.md) | Numbered error codes |

## Building

```powershell
.\scripts\check-environment.ps1               # what is missing, and the command that fixes it
dotnet build DeMosaicStudio.slnx -c Release   # 0 warnings
dotnet test  DeMosaicStudio.slnx -c Release
.\.venv\Scripts\python.exe -m pytest       # use the venv path; `python` may be the Store stub
```

The solution file is `.slnx`, not `.sln` — that is the .NET 10 default.
