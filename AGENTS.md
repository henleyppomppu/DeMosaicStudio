# Working Rules

`prd.md` says *what* to build. This says *how to work in this repository*. Where they conflict, the
PRD wins. For what is actually verified right now, see [`CLAUDE.md`](CLAUDE.md).

## Layer rules

| Project | May depend on | Must not |
| --- | --- | --- |
| `DeMosaicStudio.Domain` | nothing | touch the file system, the network, a device, or the clock |
| `DeMosaicStudio.Application` | Domain | know that the engine is a process, or that it is Python |
| `DeMosaicStudio.Infrastructure` | Domain, Application | contain policy decisions |
| `DeMosaicStudio.App` | all of the above | contain anything a Linux test needs to reach |
| `worker/demosaic_worker` | — | make UI decisions, or invent an error that has no code |
| `training/` | — | be imported by the shipped worker |

**The test for where something belongs:** if a rule cannot be tested without a GPU or a file system,
it is in the wrong place (`prd.md` §13.2). The router, the window policy, the fingerprints, the
track state machine and the error mapping are all pure and take their inputs as data — that is what
keeps CI meaningful.

## Verification

All three must pass before a commit:

```powershell
dotnet build DeMosaicStudio.slnx -c Release
dotnet test  DeMosaicStudio.slnx -c Release
.\.venv\Scripts\python.exe -m pytest
```

Note `.slnx`, not `.sln` — that is the .NET 10 default and `dotnet build …sln` fails with MSB1009.
Use the venv's interpreter explicitly; bare `python` is the Microsoft Store stub.

Platform-dependent failures will appear. Record the baseline in `CLAUDE.md` and compare against it.
**Only newly failing tests are a regression.**

## Coding rules

### C#

- Warnings are errors. Keep the build at zero.
- `Nullable` and `ImplicitUsings` are on repo-wide; do not disable them per project.
- Public types and members carry XML docs that say *why*, not what the signature already says.
- Prefer `readonly record struct` for value-like policy inputs and outputs — they are cheap, they
  print well in test failures, and `with` makes table-driven tests readable.
- Enum members must not collide with type names (`CA1720`). `GridAnchor.ObjectTracked`, not
  `Object`.

### Python

- `from __future__ import annotations` at the top of every module.
- Type hints on every public function.
- Raw strings for Windows paths (`r"C:\..."`), or `SyntaxWarning: invalid escape sequence`.
- Never print non-ASCII to stdout without calling `configure_stdio_utf8()` first — the console is
  cp949 here.
- Module docstrings explain the *reason the module exists*, not its contents.

### PowerShell

- Save `.ps1` and `.iss` as **UTF-8 with BOM**. Guarded by `ScriptEncodingTests`.
- Windows PowerShell 5.1 dialect only: no ternary, no `??`, no `&&`/`||`.
- Wrap *whole* pipelines: `@(@($x.items) | Where-Object { $_ })`.
- Console output stays ASCII. The file may contain `§`; `Write-Host` may not.

## Change procedures

### Protocol (`prd.md` §8)

1. Bump `PROTOCOL_VERSION` in `worker/demosaic_worker/protocol.py` — the **only** definition.
2. Change both sides.
3. Update `docs/WORKER_PROTOCOL.md`.
4. Add a round-trip test against `fixtures/protocol/`.

### Error codes (`prd.md` §10)

Add to **both** implementations, to `fixtures/parity/error_codes.json`, to `docs/ERROR_CODES.md`
and to `docs/TROUBLESHOOTING.md`. The parity tests fail if any step is missed. Codes are never
renumbered and never reused.

### Settings

Every setting goes in the `prd.md` §5.16.10 table with its fingerprint scope, or with the reason it
is excluded. `T-SETTINGS-FINGERPRINT-MAP-01` fails otherwise.

**Adding a key to a fingerprint invalidates every existing checkpoint** (§9.3). Sometimes correct,
always a cost, never an accident.

### Decisions

Anything hard to reverse gets an ADR in `docs/DECISIONS.md`: what was decided, what was rejected,
why, and the reversal cost.

## Never

- Weaken or delete a guard test. They exist because something actually broke.
- Put a number in a document, a commit message, or a report that a script did not produce.
- Claim an improvement without its noise floor (§13.5).
- Let a contaminating-licence dependency escape its interface (§4.2 R-4.2a).
- Continue past a distribution question. D-11 is a one-way door — stop and re-plan.
- Commit or push unless asked.

## Commits

Korean, narrative, saying what changed and why. The *why* is the part `git diff` cannot show.
