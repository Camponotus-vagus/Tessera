# Exporting the learned models

Tessera uses RaCo-ALIKED (keypoints and descriptors) and LightGlue (matching), split into fixed-size parts so they can run on Core ML. The files go to `Models/`, which is not tracked by git; `tools/fetch-models.sh` downloads a ready-made set instead.

| File | Runtime | Contents |
|---|---|---|
| `raco_aliked_levels_<H>x<W>_fp32.mlpackage` | Core ML (GPU) | default: score map, ranker map and ALIKED's four feature levels at their own resolution |
| `raco_select_k2048_<H>x<W>.onnx` | ONNX Runtime (CPU) | default: non-maximum suppression, top-k, sub-pixel refinement, ranker on the candidates |
| `aliked_descriptor_head.bin` | C++ (Accelerate) | default: weights of the descriptor head, evaluated only at the pixels it reads |
| `lightglue_raco_aliked_k2048_<fp16\|fp32>.mlpackage` | Core ML (GPU) | default: matcher (fp16 is the default precision) |
| `raco_aliked_dense_<H>x<W>_<fp32\|fp16>.mlpackage` | Core ML (GPU) | alternative split: full-resolution feature map (128 channels) |
| `raco_aliked_sparse_k2048_<H>x<W>.onnx` | ONNX Runtime (CPU) | alternative split: keypoint selection and descriptor head |
| `raco_aliked_extractor_k2048_<H>x<W>.onnx` | ONNX Runtime (CPU) | whole extractor, used when Core ML is turned off |
| `lightglue_raco_aliked_k2048.onnx` | ONNX Runtime (CPU) | matcher, CPU fallback |

`<H>x<W>` is 768x1024 (landscape photos) or 1024x768 (portrait).

## Environments

Two virtual environments are needed: Homebrew's PyTorch only exists for Python 3.14, and coremltools 9.0 only ships compiled libraries up to Python 3.13.

- `.venv`: Homebrew Python 3.14 with Homebrew's PyTorch and torchvision linked through `.pth` files, plus onnx, onnxscript and onnxruntime from PyPI.
- `.venv-coreml`: Homebrew Python 3.13 with torch, torchvision, coremltools, onnx, onnxscript and onnxruntime from PyPI.

```bash
brew install pytorch torchvision python@3.13
cd tools/export
uv venv --python /opt/homebrew/bin/python3.14 --system-site-packages .venv
echo /opt/homebrew/opt/pytorch/libexec/lib/python3.14/site-packages > .venv/lib/python3.14/site-packages/homebrew-pytorch.pth
echo /opt/homebrew/opt/torchvision/libexec/lib/python3.14/site-packages > .venv/lib/python3.14/site-packages/homebrew-torchvision.pth
VIRTUAL_ENV=.venv uv pip install onnx onnxscript onnxruntime
uv venv --python /opt/homebrew/bin/python3.13 .venv-coreml
VIRTUAL_ENV=.venv-coreml uv pip install torch torchvision coremltools onnx onnxscript onnxruntime
```

The weights (RaCo, ALIKED-n16, LightGlue for RaCo-ALIKED) are downloaded by `torch.hub` into `weights/`. `make-models.sh` clones LightGlue-ONNX at commit `d12b4ba`, which the export scripts were written against.

## Commands

```bash
./make-models.sh
```

Checks and measurements, from the repository root:

- `tools/export/.venv/bin/python tools/export/split_extractor.py check`: the split reproduces the original extractor.
- `tools/export/.venv-coreml/bin/python tools/export/bench_extractor.py <A> <B>`: Core ML extractor against ONNX.
- `tools/export/.venv-coreml/bin/python tools/export/bench_coreml.py <A> <B>`: Core ML matcher on each compute unit.

## Changes needed for Core ML

- coremltools 9.0 calls `int()` on one-element arrays, which NumPy 2.5 rejects; `convert_coreml.py` replaces the `aten::Int` conversion.
- `log_sigmoid` has no converter; it is registered as `-softplus(-x)`.
- LightGlue's self-attention builds a rank-6 tensor (Core ML allows rank 5). It is rewritten to rotate q and k separately, and checked against the original.
- RaCo's boundary ranker evaluates the ranker on patches around the candidates. On the GPU the full ranker map is computed and sampled instead, with identical results.
- The matcher returns, for every keypoint of the first image, its partner in the second and a confidence (0 when the pair is not mutual), instead of a variable-length match list.
- In ALIKED's descriptor head, the convolutions (3x3 on patches, 1x1 on samples) are written as matrix products, which ONNX Runtime runs much faster.
- The default split does not produce the 128-channel full-resolution feature map (400 MB): Core ML returns the four feature levels, and C++ rebuilds the bilinear (align_corners) upsampling and normalisation at the pixels the head reads. Checked in NumPy (difference 3.5e-6) and by a Swift test against the alternative split (identical keypoints, descriptors within 5e-6).
