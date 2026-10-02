"""Time dense (Core ML) + sparse (ONNX CPU) extraction against the all-CPU ONNX extractor.

usage: bench_extractor.py IMAGE_A IMAGE_B
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort

from bench_coreml import matches, timed
from check_split import MODELS, load, normalise

H, W, K = 768, 1024, 2048


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def geometric(kp_a, kp_b, pairs) -> set[tuple[int, int, int, int]]:
    """Matches as rounded coordinates, so different keypoint orders can be compared."""
    return {(round(float(kp_a[0, i, 0])), round(float(kp_a[0, i, 1])), round(float(kp_b[0, j, 0])),
             round(float(kp_b[0, j, 1]))) for i, j in pairs}


def main() -> None:
    a, b = load(Path(sys.argv[1]), H, W), load(Path(sys.argv[2]), H, W)
    full = session(MODELS / f"raco_aliked_extractor_k{K}_{H}x{W}.onnx")
    sparse = session(MODELS / f"raco_aliked_sparse_k{K}_{H}x{W}.onnx")
    matcher = session(MODELS / f"lightglue_raco_aliked_k{K}.onnx")

    def match(kp_a, d_a, kp_b, d_b):
        inputs = {"keypoints": np.concatenate([normalise(kp_a, H, W), normalise(kp_b, H, W)]).astype(np.float32),
                  "descriptors": np.concatenate([d_a, d_b]).astype(np.float32)}
        return matches(*matcher.run(None, inputs))

    t_full, (kp_a, d_a) = timed(lambda: full.run(None, {"image": a}), repeats=3)
    kp_b, d_b = full.run(None, {"image": b})
    reference = match(kp_a, d_a, kp_b, d_b)
    reference_xy = geometric(kp_a, kp_b, reference)
    print(f"ONNX CPU (completo)           {t_full * 1000:6.0f} ms/immagine  match {len(reference)}")

    for precision in ("fp16", "fp32"):
        for label, units in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE),
                             ("tutte", ct.ComputeUnit.ALL)]:
            dense = ct.models.MLModel(str(MODELS / f"raco_aliked_dense_{H}x{W}_{precision}.mlpackage"),
                                      compute_units=units)

            def extract(image):
                out = dense.predict({"image": image})
                return sparse.run(None, {name: out[name].astype(np.float32) for name in ("logits", "ranker", "features")})

            t_dense, _ = timed(lambda: dense.predict({"image": a}), repeats=3)
            t_total, (ka, da) = timed(lambda: extract(a), repeats=3)
            kb, db = extract(b)
            found = geometric(ka, kb, match(ka, da, kb, db))
            iou = len(found & reference_xy) / max(len(found | reference_xy), 1)
            same = np.mean(np.linalg.norm(ka[0][:, None] - kp_a[0][None], axis=2).min(axis=1) < 0.5)
            print(f"Core ML {precision} {label:5s} + sparse CPU  {t_total * 1000:6.0f} ms/immagine "
                  f"(densa {t_dense * 1000:4.0f} ms)  match {len(found)}  IoU {iou:.3f}  punti identici {same:.0%}")


if __name__ == "__main__":
    main()
