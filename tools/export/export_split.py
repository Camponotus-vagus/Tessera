"""Export RaCo-ALIKED (extractor) and LightGlue (matcher) as separate static-shape ONNX models.

The released LightGlue-ONNX pipeline runs extractor and matcher together with dynamic shapes,
which CoreML cannot take and which re-extracts every image for every pair. Here each image is
extracted once at a fixed size, and the matcher returns fixed-size outputs (best partner and
mutual score per keypoint of image A) instead of a variable-length match list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch
from onnxscript import opset20 as onnx_op

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "LightGlue-ONNX"))

from lightglue_dynamo.config import Extractor, RankerMode  # noqa: E402
from lightglue_dynamo.models import LightGlue, RaCoALIKED  # noqa: E402


class Extract(torch.nn.Module):
    def __init__(self, extractor: RaCoALIKED) -> None:
        super().__init__()
        self.extractor = extractor

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image: [1, 3, H, W] RGB in [0, 1] -> keypoints [1, K, 2] (pixels), descriptors [1, K, 128]
        return self.extractor.extract_for_matching(image)


class Match(torch.nn.Module):
    def __init__(self, matcher: LightGlue) -> None:
        super().__init__()
        self.matcher = matcher

    def forward(self, keypoints: torch.Tensor, descriptors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # keypoints: [2, K, 2] normalised by the long edge; descriptors: [2, K, 128]
        m = self.matcher
        features = m.input_proj(descriptors)
        encodings = m.posenc(keypoints)
        for layer in range(m.n_layers):
            features = m.transformers[layer](features, encodings)
        scores = m.log_assignment[m.n_layers - 1](features)  # [1, K, K]
        best0 = scores.max(2)
        best1 = scores.max(1)
        indices = torch.arange(best0.indices.shape[1], device=scores.device).expand_as(best0.indices)
        mutual = indices == best1.indices.gather(1, best0.indices)
        confidence = torch.where(mutual, best0.values.exp(), torch.zeros_like(best0.values))
        return best0.indices, confidence  # [1, K] int64, [1, K] float (0 when not mutual)


def integer_div(self: object, other: object, rounding_mode: str | None = None) -> object:
    # Same translation as the upstream CLI: indices are non-negative, so integer Div is exact.
    if rounding_mode not in {"floor", "trunc"}:
        raise ValueError(f"Unsupported integer division mode: {rounding_mode}")
    return onnx_op.Div(self, other)


def export(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], names_in: list[str], names_out: list[str],
           path: Path) -> None:
    torch.onnx.export(
        module, inputs, str(path), input_names=names_in, output_names=names_out, opset_version=20,
        dynamo=True, external_data=False, optimize=False,
        custom_translation_table={torch.ops.aten.div.Tensor_mode: integer_div},
    )
    onnx.checker.check_model(str(path))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--long-side", type=int, default=1024)
    parser.add_argument("--short-side", type=int, default=768)
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "Models")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    k = args.keypoints

    with torch.no_grad():
        for height, width in [(args.short_side, args.long_side), (args.long_side, args.short_side)]:
            mode = RankerMode.auto.resolve(k, height, width)
            extractor = RaCoALIKED(num_keypoints=k, portable_deform_conv=True, ranker_mode=mode).eval()
            extractor.fuse_batch_norm()
            print(f"extractor {height}x{width}, K={k}, ranker {mode}")
            export(Extract(extractor), (torch.zeros(1, 3, height, width),), ["image"], ["keypoints", "descriptors"],
                   args.output / f"raco_aliked_extractor_k{k}_{height}x{width}.onnx")

        matcher = LightGlue(**Extractor.raco_aliked.lightglue_config).eval()
        export(Match(matcher), (torch.rand(2, k, 2) * 2 - 1, torch.randn(2, k, 128)), ["keypoints", "descriptors"],
               ["match0", "confidence0"], args.output / f"lightglue_raco_aliked_k{k}.onnx")


if __name__ == "__main__":
    main()
