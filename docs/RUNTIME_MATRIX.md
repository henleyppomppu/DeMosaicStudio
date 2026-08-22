# Runtime Matrix

Pinned per release, as `prd.md` §6.3 requires. The product does not claim "CUDA 12.x" in general; it
claims what was verified on the machine in §4.5.

## Verified 2026-08-22

| Component | Version | Verified how |
| --- | --- | --- |
| OS | Windows 11 Pro 26200 | development machine |
| GPU | NVIDIA RTX 3080 Ti, 12 GB, compute capability 8.6 | `nvidia-smi`, `torch.cuda.get_device_capability` |
| Driver | 591.86 | `nvidia-smi` |
| .NET SDK | 10.0.302 | `dotnet --list-sdks` |
| Python | 3.12.10 | `.venv\Scripts\python.exe --version` |
| PyTorch | 2.6.0+cu124 | **a real matmul was executed on the device** — not merely a device query (§8.3) |
| torchvision | 0.21.0+cu124 | RAFT-small loaded and run |
| PyAV | 14.0.1 | decode/encode round trip with PTS assertions |
| numpy | 2.2.1 | |
| Pillow | 12.3.0 | |
| FFmpeg (`tools/ffmpeg`) | N-126239-g88ae625e69, 2026-08-21 build | `ffmpeg -version` |
| FFmpeg flags | `--enable-gpl --enable-libx264 --enable-libx265 --enable-ffnvcodec` | D-12 requires libx265 |

## Encoder availability — two FFmpeg builds, and they differ

| Encoder | `tools/ffmpeg` | PyAV's bundled FFmpeg |
| --- | --- | --- |
| libx264 | yes | yes |
| libx265 | yes | yes |
| h264_nvenc / hevc_nvenc | **yes** | **no** |

**PyAV bundles its own FFmpeg and it has no NVENC.** The Quality profile (x265) runs in-process; the
Speed profile has to shell out to `tools/ffmpeg/bin/ffmpeg.exe`. This is not what D-08 assumed, and
it is why the media layer needs both paths.

## Not supported

- **Non-NVIDIA GPUs.** No CUDA path, and DirectML was dropped under D-09/D-11. CPU only, and the UI
  states that before a job starts (§5.17c).
- **ONNX / TensorRT.** Dropped under D-09: with one target machine there is nothing to deploy to.
- **Windows 10.** Expected to work at build 19041+; never tested.

## Update procedure

1. Change the version here **and** in the corresponding lockfile (`worker/requirements.lock` for the
   engine, `training/requirements.lock` for training).
2. Re-run `scripts\check-environment.ps1`.
3. Re-run both test suites.
4. If the change touches inference, re-run §13.6's numerical comparison (CUDA FP16 vs CPU FP32)
   **before** trusting any quality metric measured afterwards. A precision regression looks exactly
   like a model regression.
