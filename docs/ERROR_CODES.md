# Error Codes

Authoritative table: `prd.md` §10.2. This file is the operator-facing copy.

Implementations, which must agree:

- `src/DeMosaicStudio.Domain/Diagnostics/ErrorCodes.cs`
- `worker/demosaic_worker/errors.py`

Locked by `fixtures/parity/error_codes.json` and the parity tests on both sides (`prd.md` §13.4).

## Adding a code

1. Add it to **both** implementations.
2. Add it to `fixtures/parity/error_codes.json`.
3. Add it here and to `docs/TROUBLESHOOTING.md`.
4. Run both test suites. The parity tests fail if any step is missed.

Codes are never renumbered and never reused.

## Rules

- Every failure crossing the host/worker boundary carries a numbered code. A free-text-only error is
  a defect (§10.1).
- `recoverable = true` means the host may auto-retry **once** with the mitigation implied by the
  code, resuming from the checkpoint rather than restarting (§10.3).
- `W`-prefixed entries are warnings. They never fail a job, and they are never marked recoverable —
  "may the host retry it" is not a question that applies to something that did not fail.

## Table

See `prd.md` §10.2 for the full table with meanings. Summary of the ranges:

| Range | Area |
| --- | --- |
| E1xxx | Media / input |
| E2xxx | Decode |
| E3xxx | Detection / tracking |
| E4xxx | Restoration (E44xx: GPU memory and backend) |
| E5xxx | Encode / mux |
| E6xxx | System (disk, permissions, RAM, support libraries) |
| E7xxx | Protocol / worker process |
| E9xxx | Unexpected internal |
| W#### | Warnings |

Warnings worth knowing about:

| Code | Meaning |
| --- | --- |
| W4102 | A region was **left untouched** because its restoration confidence fell below `minRestorationConfidence` (§5.8.1). The output at that region is the original source pixels |
| W4103 | A requested `temporalWindow` was reduced by a safety rule (scene cut, object-anchored grid, VRAM ladder, stream boundary). Carries the rule and the effective K (§5.6.1) |
| W6101 | A backend or precision was substituted — e.g. a saved `float16` re-resolved for the CPU path (§5.17b) |
