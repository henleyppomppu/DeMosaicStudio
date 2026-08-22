# Worker Protocol v1.0

Authoritative specification: `prd.md` §8. This file is the working reference that expands it.

Implementations:

- Worker (single source of `PROTOCOL_VERSION`): `worker/demosaic_worker/protocol.py`
- Host mirror: `src/DeMosaicStudio.Domain/Protocol/ProtocolVersion.cs`

Locked by `ProtocolVersionTests.The_host_mirror_matches_the_worker_definition`, which reads the
Python source rather than trusting a second constant.

## Status

**Message schemas are specified; the dispatch loop is not implemented.** Phase 0 has the version
handshake types and the stdio encoding contract. `main_loop.py` arrives with Phase 3.

## Transport

| Property | Value |
| --- | --- |
| Channel | Child process stdin (host → worker) and stdout (worker → host) |
| Encoding | **UTF-8, forced by the worker** — see below |
| Framing | One JSON object per line, `\n`-terminated, no embedded raw newlines |
| stderr | Free-form log text. Captured to the job log, **never parsed for control flow** |
| Pixel data | Never inline. Written to the job directory, referenced by path (§8.6) |

### The encoding trap

`sys.stdout.encoding` on the development machine is **cp949**, not UTF-8, because that is the
console code page for a Korean Windows locale. A worker that inherits it corrupts or crashes on the
first non-ASCII path or log message.

So the worker calls `demosaic_worker.stdio.configure_stdio_utf8()` **before the handshake**:

- stdout → UTF-8, `errors="strict"`, line-buffered. A character that cannot be encoded is a bug
  worth failing on.
- stderr → UTF-8, `errors="replace"`. Losing a diagnostic while diagnosing something else is a bad
  trade.

Guarded by `worker/tests/test_stdio.py`, which round-trips Hangul and Japanese through a real
subprocess. **A UTF-8-locale CI would never catch this.**

## Envelope

Every message:

```json
{ "v": "1.0", "type": "…", "id": "<uuid>", "jobId": "<uuid|null>" }
```

## Versioning

- The host refuses a worker whose **major** version differs, and reports `E7001`.
- Minor differences are accepted.
- **Unknown fields are ignored, never rejected.** This is what lets a newer worker add fields
  without invalidating an older host's checkpoints.

Changing the protocol follows `AGENTS.md`: bump the version → change both sides → update this file
→ add a round-trip test against `fixtures/protocol/`.

## Messages

### Host → worker

| type | Purpose | Key fields |
| --- | --- | --- |
| `hello` | Handshake | `hostVersion`, `protocolVersion` |
| `probe` | Media and hardware inspection, no processing | `sourcePath` |
| `analyze` | Detection and tracking only; produces the region summary for §5.2.5c | `jobId`, `sourcePath`, `settings`, `sampleEvery` |
| `process` | Full pipeline | `jobId`, `sourcePath`, `outputPath`, `settings`, `resume`, `comparisonPts[]` |
| `preview` | Render one frame, original and restored | `jobId`, `pts`, `settings`, `overlay` |
| `pause` / `resume` | Suspend / continue | `jobId` |
| `cancel` | Cooperative cancel | `jobId` |
| `shutdown` | Terminate the worker | — |

### Worker → host

| type | Purpose | Key fields |
| --- | --- | --- |
| `ready` | Handshake reply | `workerVersion`, `protocolVersion`, `capabilities` |
| `probeResult` | Media and hardware facts | `media{…}`, `hardware{…}` |
| `progress` | Bounded-rate progress | `stage`, `pts`, `fraction`, `fps`, `eta` |
| `log` | Structured log line | `level`, `code?`, `message`, `context{}` |
| `trackUpdate` | Diagnostics and overlay data | `frames[{pts, regions[…]}]` |
| `checkpoint` | Checkpoint written | `lastCompletedPts`, `path` |
| `previewResult` | Rendered preview | `pts`, `originalPath`, `restoredPath`, `regions[]` |
| `result` | **Terminal for a job** | `status: completed\|cancelled\|failed`, `summary{}`, `error?` |
| `error` | Failure detail | `code`, `recoverable`, `message`, `context{}` |

## Progress contract

The rules exist because out-of-order and post-terminal progress is a known failure mode of this
exact application shape — a finished job resurrecting into "processing 65%".

- At most **4 messages per second per job**, coalesced. Advisory only.
- `fraction` is monotonically non-decreasing within a job.
- `stage` moves forward only, along
  `probing → analyzing → restoring → encoding → muxing → finalizing`.
- **The host drops** any progress arriving after a terminal `result`, and any whose stage or
  fraction moves backwards. It does not apply them and does not error.
- The host's UI progress channel is ordered and inline — never a free-threaded callback that can
  reorder.

## Lifecycle

1. Host launches the worker, sends `hello`, expects `ready` within **10 s** or fails `E7002`.
2. One job at a time per worker. A second `process` is rejected with `E7003`.
3. `cancel` → acknowledge, drain, write a checkpoint, emit terminal `result` with
   `status="cancelled"` within **5 s**. After a **10 s** grace period the host may kill the process
   and record `E7004`.
4. Worker crash (non-zero exit without a terminal `result`) fails only the current job with
   `E7005`; the host relaunches for the next job.
5. `shutdown` → exit 0 within 5 s.

## Job directory

```text
%LOCALAPPDATA%\DeMosaicStudio\jobs\<jobId>\
    job.json                       # prd.md §9
    preview\<pts>.orig.png         # interactive preview, deleted on job close
    preview\<pts>.rest.png
    preview\compare\<pts>.orig.png # comparison points (§5.16.11), retained with the job
    preview\compare\<pts>.rest.png
    diag\<pts>.mask.png            # only when the diagnostic overlay is enabled
    output.tmp.<ext>
    worker.log
```

## Honest capability reporting

`probeResult.hardware.cudaAvailable` means **"we loaded the libraries and ran a test kernel"**, not
"a driver reports a device". A device count from a driver query is not availability: models load
and *then* fail. The probe exercises the path it reports on.
