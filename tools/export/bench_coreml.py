"""Time the Core ML matcher on each compute unit and compare its matches with the ONNX matcher.

usage: bench_coreml.py IMAGE_A IMAGE_B [--keypoints 2048]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort

from check_split import MODELS, load, normalise

UNITS = {
    "CPU": ct.ComputeUnit.CPU_ONLY,
    "CPU+GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU+ANE": ct.ComputeUnit.CPU_AND_NE,
    "tutte": ct.ComputeUnit.ALL,
}


def matches(match0: np.ndarray, confidence0: np.ndarray, threshold: float = 0.1) -> set[tuple[int, int]]:
    return {(i, int(match0[0, i])) for i in range(match0.shape[1]) if confidence0[0, i] > threshold}


def timed(run, repeats: int = 5) -> tuple[float, object]:
    result = run()
    start = time.perf_counter()
    for _ in range(repeats):
        result = run()
    return (time.perf_counter() - start) / repeats, result


def device_share(path: Path, units: ct.ComputeUnit) -> str:
    """Share of operations Core ML plans to run on each device."""
    try:
        compiled = ct.models.utils.compile_model(str(path))
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(path=compiled, compute_units=units)
        program = plan.model_structure.program
        counts: dict[str, int] = {}
        for function in program.functions.values():
            for operation in function.block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                if usage is None:
                    continue
                name = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
                counts[name] = counts.get(name, 0) + 1
        total = sum(counts.values()) or 1
        return ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in sorted(counts.items()))
    except Exception as error:  # noqa: BLE001
        return f"piano non disponibile ({type(error).__name__})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--keypoints", type=int, default=2048)
    args = parser.parse_args()
    k, height, width = args.keypoints, 768, 1024

    options = ort.SessionOptions()
    options.log_severity_level = 3
    extractor = ort.InferenceSession(str(MODELS / f"raco_aliked_extractor_k{k}_{height}x{width}.onnx"), options,
                                     providers=["CPUExecutionProvider"])
    kp_a, desc_a = extractor.run(None, {"image": load(args.a, height, width)})
    kp_b, desc_b = extractor.run(None, {"image": load(args.b, height, width)})
    keypoints = np.concatenate([normalise(kp_a, height, width), normalise(kp_b, height, width)]).astype(np.float32)
    descriptors = np.concatenate([desc_a, desc_b]).astype(np.float32)

    onnx_matcher = ort.InferenceSession(str(MODELS / f"lightglue_raco_aliked_k{k}.onnx"), options,
                                        providers=["CPUExecutionProvider"])
    t_onnx, (m0, c0) = timed(lambda: onnx_matcher.run(None, {"keypoints": keypoints, "descriptors": descriptors}))
    reference = matches(m0, c0)
    print(f"ONNX Runtime CPU            {t_onnx * 1000:6.0f} ms  match {len(reference)}")

    for precision in ("fp32", "fp16"):
        path = MODELS / f"lightglue_raco_aliked_k{k}_{precision}.mlpackage"
        for label, units in UNITS.items():
            model = ct.models.MLModel(str(path), compute_units=units)
            t, out = timed(lambda: model.predict({"keypoints": keypoints, "descriptors": descriptors}))
            found = matches(out["match0"].astype(np.int64), out["confidence0"])
            iou = len(found & reference) / max(len(found | reference), 1)
            print(f"Core ML {precision} {label:8s}      {t * 1000:6.0f} ms  match {len(found)}  IoU {iou:.3f}  "
                  f"[{device_share(path, units)}]")


if __name__ == "__main__":
    main()
