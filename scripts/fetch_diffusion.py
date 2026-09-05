"""Downloads a user-chosen diffusion model or LoRA from Hugging Face into the model store.

The application does not bundle a diffusion model and never will: the user names one, this fetches
it, and the store records what arrived. This script is the same code path the application will
use once the setting exists (D-44); until then it is how a model gets here.

    models/diffusion/<name>/
        <the repository's files, fp16 where the repository offers it>
        metadata.json     source, revision, file list with sizes and sha256, licence text pointer

**What this does not do.** It does not hold a Hugging Face token: a gated repository needs the
user's own login (`huggingface-cli login`) or ``HF_TOKEN`` in the environment, both of which the
hub library reads itself. It does not pick files by a hard-coded list: the repository's own file
listing is filtered by pattern, because a hard-coded list of someone else's repository is how
three 404s happened in the sibling project.

**prd.md section 2.3 C-4 applies to what is chosen.** An embedding or LoRA trained to reproduce a
particular person is identity-directed and out of scope, permanently. The tool cannot tell; the
person choosing can.

Usage::

    .venv/Scripts/python.exe scripts/fetch_diffusion.py stable-diffusion-v1-5/stable-diffusion-v1-5
    .venv/Scripts/python.exe scripts/fetch_diffusion.py latent-consistency/lcm-lora-sdv1-5 --kind lora
    .venv/Scripts/python.exe scripts/fetch_diffusion.py <repo> --name my-model --revision main

Prints ASCII only: the console this runs in is cp949.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STORE = REPO / "models" / "diffusion"

#: What to take from a diffusers-layout repository. fp16 safetensors where they exist, the
#: configuration and tokenizer files always, nothing else - a full SD1.5 repository is 40 GB of
#: every variant ever uploaded, of which about 2 GB is the model.
PIPELINE_PATTERNS = [
    "model_index.json",
    "*/config.json",
    "*/*.json",
    "*/*.txt",
    "*/*.fp16.safetensors",
    "*/diffusion_pytorch_model.safetensors",   # some repositories have no fp16 variant
    "*/model.safetensors",
]
#: A LoRA or a textual-inversion embedding is one or two files.
LORA_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.json", "README.md"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prune_duplicates(root: Path) -> None:
    """Where both an fp16 and an fp32 weight file landed, keep the fp16 one."""
    for fp16 in root.rglob("*.fp16.safetensors"):
        full = fp16.with_name(fp16.name.replace(".fp16", ""))
        if full.exists():
            full.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fetch a diffusion model or LoRA from Hugging Face")
    parser.add_argument("repo", help="Hugging Face repository id, e.g. stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--kind", choices=["pipeline", "lora", "embedding"], default="pipeline")
    parser.add_argument("--name", help="directory name under models/diffusion (default: from the repo id)")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed: pip install diffusers transformers accelerate", file=sys.stderr)
        return 2

    name = args.name or args.repo.split("/")[-1]
    destination = STORE / name
    metadata = destination / "metadata.json"
    if metadata.exists() and not args.force:
        print("already installed:", destination)
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    patterns = PIPELINE_PATTERNS if args.kind == "pipeline" else LORA_PATTERNS

    print("fetching", args.repo, "->", destination)
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        allow_patterns=patterns,
        local_dir=str(destination),
    )
    if args.kind == "pipeline":
        _prune_duplicates(destination)

    files = []
    total = 0
    for path in sorted(p for p in destination.rglob("*") if p.is_file() and p.name != "metadata.json"):
        if ".cache" in path.parts:
            continue
        size = path.stat().st_size
        total += size
        files.append({
            "path": path.relative_to(destination).as_posix(),
            "bytes": size,
            "sha256": _sha256(path) if size > 1_000_000 else None,   # hash the weights, not the JSON
        })

    metadata.write_text(json.dumps({
        "id": name,
        "kind": args.kind,
        "source": {"hub": "huggingface", "repo": args.repo, "revision": args.revision or "default"},
        "fetchedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": total,
        "files": files,
        "notes": (
            "User-chosen. Licence is the repository's own; personal use under D-11. "
            "prd.md 2.3 C-4: identity-directed embeddings/LoRAs are out of scope."
        ),
    }, indent=2) + "\n", encoding="utf-8")

    print("installed %d files, %.2f GB: %s" % (len(files), total / 2**30, destination))
    return 0


if __name__ == "__main__":
    sys.exit(main())
