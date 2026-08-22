# Architecture

`prd.md` §3 is the specification. This is the working map: what exists, what it does, and what has
not been written yet.

## Process topology

```
DeMosaicStudio.App (WPF, net10.0-windows)          [NOT BUILT]
        |
        |  stdio JSON Lines, protocol v1.0  (docs/WORKER_PROTOCOL.md)
        v
demosaic_worker (Python 3.12, .venv)
```

D-01: the boundary is a process boundary plus a wire protocol, not a language binding. A native C++
engine speaking the same protocol would be a drop-in replacement, which is why the split is drawn
here and not at a C ABI.

## What exists

| Component | State |
| --- | --- |
| `src/DeMosaicStudio.Domain` | **Built.** Error codes, protocol version mirror, settings, fingerprints, window policy, confidence gate, restoration router, track state machine |
| `src/DeMosaicStudio.Application` | **Does not exist** |
| `src/DeMosaicStudio.Infrastructure` | **Does not exist** |
| `src/DeMosaicStudio.App` | **Does not exist** |
| `worker/demosaic_worker/protocol.py` | Message and version types. **No dispatch loop yet** |
| `worker/demosaic_worker/errors.py` | The §10 table |
| `worker/demosaic_worker/fingerprints.py` | §9.3 canonical form and invalidation |
| `worker/demosaic_worker/stdio.py` | UTF-8 enforcement (the cp949 trap) |
| `worker/demosaic_worker/media/` | Probe, PTS-preserving passthrough, timing rules |
| `worker/demosaic_worker/{scene,detect,track,analyze,roi,window,restore,post}` | **Do not exist** |
| `training/` | Degradation generator, motion estimation, metrics, IBP solver, dense flow, U-Net, dataset, training loop |

`training/` is not part of the shipped engine and is never imported by `demosaic_worker`
(AGENTS.md layer rules).

## Domain — where the decisions live

Everything that decides something is pure, takes its inputs as data, and has no device or file
dependency. That is what makes the GPU-free test subset meaningful (§13.2) and it is the rule to
check first when deciding where new code goes.

| Type | Decides | prd.md |
| --- | --- | --- |
| `TemporalWindowPolicy` | how many neighbours to fuse, and why fewer than asked | §5.6, §5.6.1 |
| `RestorationRouter` | multi-frame / single-frame / pass-through, with a reason from a closed enum | §5.8 |
| `ConfidenceGate` | whether to withhold a restoration and keep the original pixels | §5.8.1 |
| `SettingsFingerprint` + `ArtifactInvalidation` | what a resume may reuse | §9.3 |
| `TrackStateMachine` | legal track transitions; illegal ones raise E3201 | §5.3.3 |
| `ErrorCodes` | the numbered failure vocabulary | §10 |

Each has a Python mirror where the worker needs the same semantics, locked by fixtures in
`fixtures/parity/` (§13.4).

## Data flow, as specified

```
Input video
   |
FFmpeg demux ──> audio / subtitles ──────────────┐ (stream copy, never transcoded)
   |                                             |
Decode (PyAV)                                    |
   |                                             |
PTS-ordered scheduler ─┬─ scene cut              |
   |                   └─ mosaic segmentation    |
   |                          |                  |
   |                     ByteTrack + Kalman      |
   |                          |                  |
   |              MosaicProfile + grid anchoring |
   |                          |                  |
   |                     ROI stabiliser          |
   |                          |                  |
   |                  Adaptive window (K)        |
   |                          |                  |
   |                  Dense flow alignment       |
   |                          |                  |
   |                  Restoration router         |
   |               ┌──────────┼──────────┐       |
   |          multi-frame  single    pass-through|
   |               └──────────┼──────────┘       |
   |                  Temporal consistency       |
   |                          |                  |
   |                  Mask-aware blending        |
   |                          |                  |
Encode (x265 / NVENC) ────────┘                  |
   |                                             |
Mux <────────────────────────────────────────────┘
   |
Output video (+ synthetic / confidence metadata)
```

Of that diagram, **decode → passthrough → encode → mux is implemented and tested**; the stages
between are not.

## Measurements that shaped the design

Three experiments changed the plan. Their reports carry the numbers and the limitations.

| Finding | Consequence | Report |
| --- | --- | --- |
| Multi-frame recovers +3.30 dB with perfect alignment; the information is there | Phase 2 not killed | `phase0-report.md` |
| Object-anchored grids lose 0.79–1.50 dB even with perfect alignment | Grid-anchoring detection is a safety gate, not a diagnostic | `phase0-report.md` |
| Dense flow closes only 19% of the alignment gap; static content has the *best* alignment and no gain | The limit is content correspondence, not flow accuracy. Multi-frame's window is screen-anchored + slow motion, K=3 | `phase2-alignment-report.md` |
| A 7.8 M U-Net reaches 0.82 held-out IoU on 96 s of video | The corpus is not the detector's limit; false positives are | `phase1-detector-report.md` |

## Known limits

- **No real mosaicked footage has ever been processed.** Every positive sample so far is synthetic.
- Everything measured used a classical solver. Whether a learned restorer tolerates imperfect
  frame-to-frame correspondence is Phase 2's open question and is what decides multi-frame's scope.
- One target machine (§4.5), one source film, PSNR only — no LPIPS, no warping error, no human
  assessment of any output.

Empty placeholder directories are deliberately absent rather than committed empty: git does not track
them, so a fresh clone would show them missing anyway, and a placeholder that looks like a project
but has no `.csproj` reads as "started" when nothing was.
