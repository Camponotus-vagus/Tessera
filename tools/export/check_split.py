"""Check the split models against the released pipeline and time them on CPU and CoreML.

usage: check_split.py IMAGE_A IMAGE_B [--keypoints 2048]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODELS = Path(__file__).resolve().parents[2] / "Models"


def load(path: Path, height: int, width: int) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((width, height), Image.LANCZOS)
    return (np.asarray(image, dtype=np.float32) / 255).transpose(2, 0, 1)[None]  # [1, 3, H, W]


def session(path: Path, provider: str, units: str | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers: list = ["CPUExecutionProvider"]
    if provider == "coreml":
        providers = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": units,
                                                  "RequireStaticInputShapes": "1"}), "CPUExecutionProvider"]
    return ort.InferenceSession(str(path), options, providers=providers)


def normalise(keypoints: np.ndarray, height: int, width: int) -> np.ndarray:
    size = np.array([width, height], dtype=np.float32)
    return (keypoints - size / 2) / (size.max() / 2)


def timed(run, repeats: int = 5) -> tuple[float, object]:
    result = run()  # warm-up (CoreML compiles on first run)
    start = time.perf_counter()
    for _ in range(repeats):
        result = run()
    return (time.perf_counter() - start) / repeats, result


def coreml_coverage(path: Path, units: str) -> str:
    """How many nodes CoreML claims, read from ORT's verbose log in a subprocess."""
    code = (
        "import onnxruntime as ort\n"
        "o = ort.SessionOptions(); o.log_severity_level = 1\n"
        f"ort.InferenceSession({str(path)!r}, o, providers=[('CoreMLExecutionProvider', "
        f"{{'ModelFormat': 'MLProgram', 'MLComputeUnits': {units!r}, 'RequireStaticInputShapes': '1'}}), "
        "'CPUExecutionProvider'])\n"
    )
    log = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stderr
    found = re.findall(r"number of partitions supported by CoreML: (\d+) number of nodes in the graph: (\d+) "
                       r"number of nodes supported by CoreML: (\d+)", log)
    if not found:
        return "n/d"
    partitions, nodes, supported = found[-1]
    return f"{supported}/{nodes} nodi in {partitions} partizioni"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--keypoints", type=int, default=2048)
    args = parser.parse_args()
    k, height, width = args.keypoints, 768, 1024
    image_a, image_b = load(args.a, height, width), load(args.b, height, width)

    # Reference: the released dynamic pipeline on CPU.
    pipeline = session(MODELS / f"raco_aliked_lightglue_pipeline_k{k}.onnx", "cpu")
    keypoints_ref, matches_ref, _ = pipeline.run(None, {"images": np.concatenate([image_a, image_b])})
    reference = {(int(m[1]), int(m[2])) for m in matches_ref if m[0] == 0}

    extractor_path = MODELS / f"raco_aliked_extractor_k{k}_{height}x{width}.onnx"
    matcher_path = MODELS / f"lightglue_raco_aliked_k{k}.onnx"
    variants = [("cpu", None), ("coreml", "CPUOnly"), ("coreml", "CPUAndGPU"), ("coreml", "CPUAndNeuralEngine"),
                ("coreml", "ALL")]
    for provider, units in variants:
        label = provider if units is None else f"coreml {units}"
        try:
            extractor = session(extractor_path, provider, units)
            matcher = session(matcher_path, provider, units)
        except Exception as error:  # noqa: BLE001 - report and keep going with the other variants
            print(f"{label:28s} non parte: {str(error).splitlines()[0][:120]}")
            continue
        t_extract, (kp_a, desc_a) = timed(lambda: extractor.run(None, {"image": image_a}))
        _, (kp_b, desc_b) = timed(lambda: extractor.run(None, {"image": image_b}), repeats=1)
        inputs = {
            "keypoints": np.concatenate([normalise(kp_a, height, width), normalise(kp_b, height, width)]),
            "descriptors": np.concatenate([desc_a, desc_b]),
        }
        t_match, (match0, confidence0) = timed(lambda: matcher.run(None, inputs))
        found = {(i, int(match0[0, i])) for i in range(k) if confidence0[0, i] > 0.1}
        drift = float(np.abs(np.concatenate([kp_a[0], kp_b[0]]) - keypoints_ref.reshape(-1, 2)).max())
        overlap = len(found & reference) / max(len(found | reference), 1)
        coverage = "" if provider == "cpu" else "  CoreML: " + "; ".join(
            coreml_coverage(p, units) for p in (extractor_path, matcher_path))
        print(f"{label:28s} estrazione {t_extract * 1000:6.0f} ms  matching {t_match * 1000:6.0f} ms  "
              f"coppia {(2 * t_extract + t_match) * 1000:6.0f} ms  match {len(found):4d} (rif. {len(reference)}) "
              f"IoU {overlap:.3f}  scarto punti {drift:.3f} px{coverage}")


if __name__ == "__main__":
    main()
