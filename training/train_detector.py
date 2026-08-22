"""Experiment A — can a detector learn the mosaic signature at all?

This is not a production training run. It answers one question, cheaply, before anyone spends money
on data: **does the architecture in D-03 learn to segment a synthetic mosaic from the corpus already
in hand?**

* If held-out IoU is poor, the problem is the architecture, and no quantity of additional video
  fixes it — D-03 gets revisited.
* If it is good, the corpus is not the current limit, and the next thing to expand is *negatives*
  (§11.4), because the false-positive requirement of §5.2.5a is what binds.

Splits are by **clip**, stratified by motion band (§11.6). Validation clips are never seen in
training, so the numbers say something about a shot the model has not looked at — though still from
the same film, which §5 of the report is explicit about.

Usage::

    .venv/Scripts/python.exe training/train_detector.py --steps 3000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))

from data.dataset import load_split, make_batch  # noqa: E402
from demosaic_worker.detect.unet import (  # noqa: E402
    MosaicUNet,
    dice_bce_loss,
    false_positive_area,
    mask_iou,
)

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "training" / "datasets" / "clean"
MANIFEST = REPO / "training" / "datasets" / "clean-tos.manifest.json"


@torch.no_grad()
def evaluate_by_block(
    model: MosaicUNet,
    clips: list,
    device: torch.device,
    *,
    seed: int = 7,
    per_block: int = 48,
) -> dict[str, dict[str, float]]:
    """IoU broken down by block size.

    An aggregate IoU hides the shape of the failure. prd.md §1.4.2 says small blocks destroy less
    and large blocks destroy more, so a detector that only works in the middle of that range is a
    different problem from one that does not work at all — and the two call for different fixes.
    """
    from data.dataset import make_sample  # local import: keeps the module importable without a corpus

    model.eval()
    buckets: dict[str, list[float]] = {}

    rng = np.random.default_rng(seed)
    for _ in range(per_block):
        images = []
        masks = []
        blocks = []

        for _ in range(8):
            image, mask, spec = make_sample(clips, rng, positive_rate=1.0)
            images.append(image.astype(np.float32) / 255.0)
            masks.append(mask)
            blocks.append(spec.block)

        x = torch.from_numpy(np.stack(images)[:, None]).to(device)
        y = torch.from_numpy(np.stack(masks)[:, None]).to(device)
        ious = mask_iou(model(x), y).cpu().tolist()

        for block, iou in zip(blocks, ious, strict=True):
            key = (
                "4-6" if block <= 6
                else "7-12" if block <= 12
                else "13-18" if block <= 18
                else "19-24"
            )
            buckets.setdefault(key, []).append(iou)

    model.train()

    return {
        key: {"n": len(values), "iou_mean": round(float(np.mean(values)), 4)}
        for key, values in sorted(buckets.items())
    }


@torch.no_grad()
def evaluate(
    model: MosaicUNet,
    clips: list,
    rng: np.random.Generator,
    device: torch.device,
    *,
    batches: int = 16,
    batch_size: int = 8,
) -> dict[str, float]:
    """Measures IoU on mosaicked crops and false-positive area on clean ones, separately.

    Two passes rather than one mixed pass: an aggregate over both would let a good IoU hide a bad
    false-positive rate, and §5.2.5 says the second is the one that damages a user's footage.
    """
    model.eval()

    ious: list[float] = []
    for _ in range(batches):
        images, masks = make_batch(clips, rng, batch_size, positive_rate=1.0)
        x = torch.from_numpy(images).to(device)
        y = torch.from_numpy(masks).to(device)
        ious.extend(mask_iou(model(x), y).cpu().tolist())

    false_positives: list[float] = []
    for _ in range(batches):
        images, masks = make_batch(clips, rng, batch_size, positive_rate=0.0, hard_negative_rate=0.5)
        x = torch.from_numpy(images).to(device)
        y = torch.from_numpy(masks).to(device)
        false_positives.extend(false_positive_area(model(x), y).cpu().tolist())

    model.train()

    fp = np.asarray(false_positives)

    return {
        "iou_mean": float(np.mean(ious)),
        "iou_median": float(np.median(ious)),
        "iou_p10": float(np.percentile(ious, 10)),
        "fp_area_mean": float(fp.mean()),
        "fp_area_p95": float(np.percentile(fp, 95)),
        "fp_crops_over_half_percent": float((fp > 0.005).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "phase1-detector.json")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        raise SystemExit(f"corpus manifest missing: {MANIFEST}. Run scripts/build_corpus.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}", flush=True)
    print("loading corpus", flush=True)
    train_clips, val_clips, names = load_split(MANIFEST, CORPUS)
    print(f"  train {len(train_clips)} clips, val {len(val_clips)} clips", flush=True)
    print(f"  val: {', '.join(names['val'])}", flush=True)

    torch.manual_seed(args.seed)
    model = MosaicUNet(width=args.width).to(device)
    print(f"parameters: {model.parameter_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.15
    )

    train_rng = np.random.default_rng(args.seed)
    # Fixed seed for evaluation so successive checkpoints are compared on the same crops. Without
    # this, the metric moves because the sample moved, and the noise floor swallows the signal.
    history: list[dict[str, float]] = []

    started = time.time()
    running = 0.0

    for step in range(1, args.steps + 1):
        images, masks = make_batch(train_clips, train_rng, args.batch_size, size=args.crop)
        x = torch.from_numpy(images).to(device)
        y = torch.from_numpy(masks).to(device)

        loss = dice_bce_loss(model(x), y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()

        running += float(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            train_metrics = evaluate(model, train_clips, np.random.default_rng(7), device)
            val_metrics = evaluate(model, val_clips, np.random.default_rng(7), device)

            entry = {
                "step": step,
                "loss": round(running / args.eval_every, 4),
                "train_iou": round(train_metrics["iou_mean"], 4),
                "val_iou": round(val_metrics["iou_mean"], 4),
                "val_iou_p10": round(val_metrics["iou_p10"], 4),
                "val_fp_area_mean": round(val_metrics["fp_area_mean"], 5),
                "val_fp_crops_over_half_percent": round(
                    val_metrics["fp_crops_over_half_percent"], 4
                ),
            }
            history.append(entry)
            running = 0.0

            print(
                f"step {step:5d}  loss {entry['loss']:.4f}  "
                f"train IoU {entry['train_iou']:.3f}  val IoU {entry['val_iou']:.3f} "
                f"(p10 {entry['val_iou_p10']:.3f})  "
                f"val FP area {entry['val_fp_area_mean']:.4f}",
                flush=True,
            )

    elapsed = time.time() - started
    final_val = evaluate(model, val_clips, np.random.default_rng(7), device, batches=32)
    final_train = evaluate(model, train_clips, np.random.default_rng(7), device, batches=32)
    by_block = evaluate_by_block(model, val_clips, device)

    checkpoint = args.out.with_suffix(".pt")
    torch.save(
        {"state_dict": model.state_dict(), "width": args.width, "steps": args.steps},
        checkpoint,
    )

    report = {
        "experiment": "A — can the detector learn the mosaic signature from the existing corpus?",
        "device": str(device),
        "parameters": model.parameter_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "crop": args.crop,
        "seconds": round(elapsed, 1),
        "splits": names,
        "final": {"train": final_train, "val": final_val},
        "val_iou_by_block": by_block,
        "checkpoint": checkpoint.name,
        "history": history,
        "not_measured": [
            "generalisation across films — every clip comes from one source (prd.md §11.2)",
            "H.264 recompression — training uses JPEG as a per-crop stand-in (§11.3)",
            "real mosaics — all positives are synthetic, which is what makes the mask exact",
            "the §5.2.5a false-positive rate proper, which needs the collected negatives corpus",
        ],
    }

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"train IoU {final_train['iou_mean']:.3f}   val IoU {final_val['iou_mean']:.3f}")
    print(f"val IoU p10 {final_val['iou_p10']:.3f}   (the worst tenth of crops)")
    print(f"val FP area {final_val['fp_area_mean']:.4f} mean, {final_val['fp_area_p95']:.4f} p95")
    print(f"val clean crops marking >0.5% of area: {final_val['fp_crops_over_half_percent']:.1%}")
    print()
    print("val IoU by block size:")
    for key, stats in by_block.items():
        print(f"  {key:>6} px   n={stats['n']:4d}   IoU {stats['iou_mean']:.3f}")
    print()
    print(f"elapsed {elapsed:.0f}s   checkpoint {checkpoint.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
