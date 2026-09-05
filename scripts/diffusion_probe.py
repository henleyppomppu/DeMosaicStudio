"""Phase 0 for a diffusion restorer: does img2img on the region beat bicubic, and does it flicker?

Takes the quality fixture (a screen-fixed mosaic over a panning crop of Tears of Steel, with the
clean frames known), cuts the region out of each frame, and restores it three ways:

* bicubic  - decimate to block resolution, interpolate back. The Fast preset; the floor.
* diffusion at several strengths - the bicubic result as the init image, a user-chosen SD1.5
  pipeline with an LCM LoRA, four steps, fixed seed, an empty prompt. Strength is the whole
  question: 0 is bicubic, 1 ignores the observation and invents.

Reports, per method: PSNR and LPIPS inside the region against the clean frame, the mean LPIPS
between consecutive *outputs* (flicker: a restorer that draws something different every frame
scores high here even when each frame alone looks fine), and seconds per region. Writes one PNG
strip of a frame through every method.

The prompt is empty on purpose and the script accepts none: prd.md section 2.3 C-4 forbids steering
a restoration toward a person, and the way to be sure a prompt does not is to have no prompt.

Usage::

    .venv/Scripts/python.exe scripts/diffusion_probe.py --model stable-diffusion-v1-5 --lora lcm-lora-sdv1-5
        [--strengths 0.3 0.5 0.7] [--steps 4] [--frames 24] [--out artifacts/diffusion-probe.png]

Model and LoRA are names under models/diffusion/ (see fetch_diffusion.py).

Prints ASCII only: the console this runs in is cp949.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))
sys.path.insert(0, str(REPO / "worker" / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

import av  # noqa: E402
import test_endtoend_quality as T  # noqa: E402
from demosaic_worker.metrics import psnr  # noqa: E402
from demosaic_worker.restore.ibp import block_average  # noqa: E402
from demosaic_worker.restore.upscale import decimate, resize  # noqa: E402

STORE = REPO / "models" / "diffusion"


def build(frames: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    mask = T._region()
    with av.open(str(T.CORPUS)) as container:
        source = [f.to_ndarray(format="rgb24")
                  for _, f in zip(range(frames), container.decode(container.streams.video[0]))]
    clean, degraded = [], []
    for index, frame in enumerate(source):
        offset = min(index * T.PAN, 120)
        picture = frame[200:200 + T.HEIGHT, 300 + offset:300 + offset + T.WIDTH].copy()
        clean.append(picture)
        blocks = np.stack([block_average(picture[:, :, c].astype(np.float64), T.SPEC, (0, 0))
                           for c in range(3)], axis=2)
        degraded.append(np.where(mask[:, :, None], blocks, picture).astype(np.uint8))
    return clean, degraded, mask


def bicubic_rgb(crop: np.ndarray) -> np.ndarray:
    """Decimate and interpolate each channel; the Fast preset on an RGB crop."""
    return np.stack([resize(decimate(crop[:, :, c].astype(np.float64), T.SPEC, (0, 0)), crop.shape[:2])
                     for c in range(3)], axis=2)


def multiple_of_8(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    H, W = -(-h // 8) * 8, -(-w // 8) * 8
    padded = np.pad(image, ((0, H - h), (0, W - w), (0, 0)), mode="edge")
    return padded, (h, w)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bicubic vs diffusion img2img on the mosaic region")
    parser.add_argument("--model", required=True, help="name under models/diffusion/")
    parser.add_argument("--lora", default=None, help="name under models/diffusion/ (LCM LoRA)")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--work-size", type=int, default=512,
                        help="long side the crop is upscaled to before img2img; SD1.5 was trained at 512")
    parser.add_argument("--prompt", default="",
                        help="generic content words only (e.g. 'photo'); never a person - prd.md 2.3 C-4")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "diffusion-probe.png")
    args = parser.parse_args(argv)

    import torch
    from diffusers import AutoPipelineForImage2Image, LCMScheduler
    from PIL import Image, ImageDraw

    from perceptual import distance

    clean, degraded, mask = build(args.frames)
    ys, xs = np.nonzero(mask)
    pad = 24
    top, left = max(0, ys.min() - pad), max(0, xs.min() - pad)
    bottom, right = min(mask.shape[0], ys.max() + 1 + pad), min(mask.shape[1], xs.max() + 1 + pad)
    region = mask[top:bottom, left:right]
    print("frames %d, region crop %dx%d, work size %d, steps %d, guidance %.1f, prompt %r"
          % (len(clean), right - left, bottom - top, args.work_size, args.steps, args.guidance, args.prompt))

    # The store keeps the fp16 variant only (fetch_diffusion.py), so the loader must ask for it.
    pipe = AutoPipelineForImage2Image.from_pretrained(
        str(STORE / args.model), torch_dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False,
    ).to("cuda")
    if args.lora:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        lora_dir = STORE / args.lora
        weight = next(lora_dir.glob("*.safetensors"))
        pipe.load_lora_weights(str(lora_dir), weight_name=weight.name)
        pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    print("pipeline loaded; VRAM %.1f GB" % (torch.cuda.memory_allocated() / 2**30))

    methods: dict[str, list[np.ndarray]] = {"bicubic": []}
    seconds: dict[str, float] = {"bicubic": 0.0}
    for s in args.strengths:
        methods["diffusion s=%.1f" % s] = []
        seconds["diffusion s=%.1f" % s] = 0.0

    for i in range(len(clean)):
        crop = degraded[i][top:bottom, left:right]
        t0 = time.perf_counter()
        base = bicubic_rgb(crop)
        seconds["bicubic"] += time.perf_counter() - t0
        methods["bicubic"].append(base)

        # Work at the network's own scale. A 200-pixel crop fed straight in is far outside what
        # SD1.5 saw in training and comes back as coloured blobs (the first run of this probe);
        # tile upscalers exist for exactly this reason. Up by an integer factor to about
        # `work_size` on the long side, run, and come back down.
        h, w = base.shape[:2]
        factor = max(1, round(args.work_size / max(h, w)))
        big = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((w * factor, h * factor), Image.BICUBIC)
        init, (H, W) = multiple_of_8(np.asarray(big))
        for s in args.strengths:
            key = "diffusion s=%.1f" % s
            generator = torch.Generator("cuda").manual_seed(args.seed)
            t0 = time.perf_counter()
            out = pipe(
                prompt=args.prompt, image=Image.fromarray(init), strength=s,
                num_inference_steps=args.steps, guidance_scale=args.guidance, generator=generator,
            ).images[0]
            torch.cuda.synchronize()
            seconds[key] += time.perf_counter() - t0
            arr = np.asarray(out.crop((0, 0, W, H)).resize((w, h), Image.LANCZOS), dtype=np.float64)
            # Only the region is the restorer's; the surroundings come back from the crop.
            methods[key].append(np.where(region[:, :, None], arr, crop))

    grey = lambda a: np.asarray(a, dtype=np.float64).mean(axis=2)  # noqa: E731
    reference = [clean[i][top:bottom, left:right] for i in range(len(clean))]
    inputs = [degraded[i][top:bottom, left:right] for i in range(len(clean))]

    def score(images: list[np.ndarray]) -> tuple[float, float, float]:
        p = np.mean([psnr(grey(reference[i])[region], grey(images[i])[region]) for i in range(len(images))])
        l = np.mean([distance(reference[i], np.clip(images[i], 0, 255).astype(np.uint8)) for i in range(len(images))])
        flicker = np.mean([
            distance(np.clip(images[i - 1], 0, 255).astype(np.uint8), np.clip(images[i], 0, 255).astype(np.uint8))
            for i in range(1, len(images))
        ])
        return float(p), float(l), float(flicker)

    print()
    print("%-18s %8s %8s %9s %9s" % ("", "PSNR", "LPIPS", "flicker", "s/region"))
    p, l, f = score(inputs)
    print("%-18s %8.2f %8.4f %9.4f %9s" % ("input (mosaic)", p, l, f, "-"))
    p, l, f = score(reference[1:] + reference[:1])  # clean vs clean shifted: the flicker floor of real motion
    print("%-18s %8s %8s %9.4f %9s" % ("(clean, motion)", "-", "-", f, "-"))
    for key, images in methods.items():
        p, l, f = score(images)
        print("%-18s %8.2f %8.4f %9.4f %9.3f" % (key, p, l, f, seconds[key] / len(images)))
    print()
    print("flicker = mean LPIPS between consecutive outputs; the clean row is what real motion alone costs.")

    i = min(20, len(clean) - 1)
    panels = [("clean", reference[i]), ("input", inputs[i])] + [(k, v[i]) for k, v in methods.items()]
    scale = 2
    tiles = []
    for label, image in panels:
        im = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).resize(
            (image.shape[1] * scale, image.shape[0] * scale), Image.NEAREST)
        canvas = Image.new("RGB", (im.width, im.height + 22), "black")
        canvas.paste(im, (0, 22))
        ImageDraw.Draw(canvas).text((4, 4), label, fill="white")
        tiles.append(canvas)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), tiles[0].height), "black")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 6
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
