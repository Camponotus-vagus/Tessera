"""Split RaCo-ALIKED into a dense part for Core ML (GPU) and a sparse part for ONNX Runtime (CPU).

dense:  image [1, 3, H, W]          -> logits [1, 1, H, W], ranker [1, 1, H, W], features [1, 128, H, W]
sparse: logits, ranker, features    -> keypoints [1, K, 2], descriptors [1, K, 128]

The dense convolutions are most of the cost. The ranker map is computed densely on the GPU and
sampled at the candidates, which is what RaCo's candidate-local ranker reproduces on the CPU.
Top-k selection and the deformable descriptor head work on a few thousand points and stay on the CPU.

usage: split_extractor.py {check,dense,sparse,levels,select,head} [--keypoints 2048]
  check   verify in PyTorch that sparse(dense(x)) equals the original extractor
  dense   write raco_aliked_dense_<H>x<W>_<precision>.mlpackage (needs .venv-coreml)
  sparse  write raco_aliked_sparse_k<K>_<H>x<W>.onnx (needs .venv)

Faster variant used by the app (no full-resolution 128-channel map):
  levels  write raco_aliked_levels_<H>x<W>_fp32.mlpackage: logits, ranker and ALIKED's four feature
          levels at their own resolution (needs .venv-coreml)
  select  write raco_select_k<K>_<H>x<W>.onnx: keypoint selection from logits and ranker (needs .venv)
  head    write aliked_descriptor_head.bin: weights of the descriptor head, evaluated in C++
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "LightGlue-ONNX"))

from lightglue_dynamo.config import RankerMode  # noqa: E402
from lightglue_dynamo.models import RaCoALIKED  # noqa: E402
from lightglue_dynamo.models.aliked import _get_patches  # noqa: E402
from lightglue_dynamo.models.raco import _chunked_topk, _gather_subpixel_offsets, _sample  # noqa: E402

SIZES = [(768, 1024), (1024, 768)]
MODELS = ROOT.parents[1] / "Models"


class Dense(torch.nn.Module):
    def __init__(self, extractor: RaCoALIKED) -> None:
        super().__init__()
        self.raco = extractor.raco
        self.aliked = extractor.aliked

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.raco
        normalised = (image - r.image_mean) / r.image_std
        x1 = r.block1(normalised)
        x2 = r.block2(r.pool2(x1))
        x3 = r.block3(r.pool4(x2))
        x4 = r.block4(r.pool4(x3))
        features = torch.cat(
            [
                r.gate(r.conv1(x1)),
                F.interpolate(r.gate(r.conv2(x2)), scale_factor=2, mode="bilinear", align_corners=True),
                F.interpolate(r.gate(r.conv3(x3)), scale_factor=8, mode="bilinear", align_corners=True),
                F.interpolate(r.gate(r.conv4(x4)), scale_factor=32, mode="bilinear", align_corners=True),
            ],
            dim=1,
        )
        return r.score_head(features), r.ranker_head(normalised), self.aliked._dense_features(image)


def descriptor_head_matmul(head: torch.nn.Module, features: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    """ALIKED's sparse deformable descriptor head with its convolutions written as matrix products.

    The convolutions act on 3x3 patches and on single samples, so they are plain GEMMs; ONNX Runtime
    runs them much faster that way than as Conv nodes on 1x1 images.
    """
    batch, channels, height, width = features.shape
    count, positions, kernel = keypoints.shape[1], head.positions, head.kernel_size
    scale = torch.tensor([width - 1, height - 1], dtype=features.dtype)
    pixel_points = (keypoints / 2 + 0.5) * scale
    patches = _get_patches(features, pixel_points.long(), kernel).reshape(batch * count, channels * kernel * kernel)
    first, activation, second = head.offset_conv
    hidden = activation(patches @ first.weight.reshape(first.out_channels, -1).T + first.bias)
    offsets = hidden @ second.weight.reshape(second.out_channels, -1).T + second.bias
    maximum = max(height, width) / 4.0
    offsets = offsets.clamp(-maximum, maximum).reshape(batch, count, 2, positions).transpose(2, 3)
    grid = (2 * (pixel_points[:, :, None] + offsets) / scale - 1).reshape(batch, count * positions, 1, 2)
    sampled = F.grid_sample(features, grid, mode="bilinear", align_corners=True)  # [B, C, N*P, 1]
    sampled = sampled.reshape(batch, channels, count * positions).transpose(1, 2)  # [B, N*P, C]
    sampled = F.selu(sampled @ head.sf_conv.weight.reshape(channels, channels).T)
    descriptors = sampled.reshape(batch, count, positions * channels) @ head.agg_weights.reshape(positions * channels, -1)
    return F.normalize(descriptors, p=2, dim=2)


class Levels(torch.nn.Module):
    """Dense half without ALIKED's upsampling, concatenation and normalisation (done per pixel in C++)."""

    def __init__(self, extractor: RaCoALIKED) -> None:
        super().__init__()
        self.dense = Dense(extractor)
        self.aliked = extractor.aliked

    def forward(self, image: torch.Tensor):
        logits, ranker, _ = self.dense(image)
        a = self.aliked
        y1 = a.block1(image)
        y2 = a.block2(a.pool2(y1))
        y3 = a.block3(a.pool4(y2))
        y4 = a.block4(a.pool4(y3))
        return logits, ranker, a.gate(a.conv1(y1)), a.gate(a.conv2(y2)), a.gate(a.conv3(y3)), a.gate(a.conv4(y4))


class Select(torch.nn.Module):
    """Keypoint selection only: the first half of Sparse."""

    def __init__(self, extractor: RaCoALIKED) -> None:
        super().__init__()
        self.sparse = Sparse(extractor)

    def forward(self, logits: torch.Tensor, ranker: torch.Tensor) -> torch.Tensor:
        return self.sparse.keypoints(logits, ranker)


class Sparse(torch.nn.Module):
    """RaCo boundary-ranked keypoint selection and the ALIKED descriptor head (RankerMode.boundary)."""

    def __init__(self, extractor: RaCoALIKED) -> None:
        super().__init__()
        if extractor.ranker_mode is not RankerMode.boundary:
            raise ValueError("the split mirrors the boundary ranker used for K = 1024...2048")
        self.raco = extractor.raco
        self.aliked = extractor.aliked
        self.num_keypoints = extractor.raco.num_keypoints

    def forward(self, logits: torch.Tensor, ranker: torch.Tensor, features: torch.Tensor):
        keypoints = self.keypoints(logits, ranker)
        height, w = logits.shape[-2], logits.shape[-1]
        scale = torch.tensor([w - 1, height - 1], dtype=keypoints.dtype)
        descriptors = descriptor_head_matmul(self.aliked.desc_head, features, 2 * keypoints / scale - 1)
        return keypoints, descriptors

    def keypoints(self, logits: torch.Tensor, ranker: torch.Tensor) -> torch.Tensor:
        r = self.raco
        # RaCo._candidate_keypoints after the score head.
        nms = F.max_pool2d(logits, r.nms_radius, stride=1, padding=r.nms_radius // 2)
        logits_nms = torch.where(logits == nms, logits, -torch.inf)
        _values, top_indices = _chunked_topk(logits_nms.flatten(1), r.num_candidates, r.topk_chunk_size)
        width = logits.shape[-1]
        x = torch.remainder(top_indices, width)
        y = torch.div(top_indices, width, rounding_mode="floor")
        keypoints = torch.stack((x, y), dim=-1).to(logits.dtype)
        if r.subpixel_sampling:
            keypoints = keypoints + _gather_subpixel_offsets(logits, top_indices, r.nms_radius,
                                                             r.subpixel_temperature)
        # RaCo.extract_boundary_ranked after _candidate_keypoints.
        window_start = self.num_keypoints - r.boundary_reranked_count
        window = keypoints[:, window_start: window_start + r.boundary_window_count]
        window_scores = _sample(ranker, window)
        window_order = window_scores.topk(r.boundary_reranked_count, dim=1).indices
        prefix = torch.arange(window_start, dtype=window_order.dtype)[None].expand(window_order.shape[0], -1)
        selected = torch.cat((prefix, window_order + window_start), dim=1)
        return keypoints.gather(1, selected[..., None].expand(-1, -1, 2)) + 0.5


def build(keypoints: int, height: int, width: int) -> RaCoALIKED:
    mode = RankerMode.auto.resolve(keypoints, height, width)
    extractor = RaCoALIKED(num_keypoints=keypoints, portable_deform_conv=True, ranker_mode=mode).eval()
    extractor.fuse_batch_norm()
    return extractor


def check(k: int) -> None:
    torch.manual_seed(0)
    for height, width in SIZES:
        extractor = build(k, height, width)
        image = torch.rand(1, 3, height, width)
        with torch.no_grad():
            reference = extractor.extract_for_matching(image)
            candidate = Sparse(extractor)(*Dense(extractor)(image))
        dk = (reference[0] - candidate[0]).abs().max().item()
        dd = (reference[1] - candidate[1]).abs().max().item()
        print(f"{height}x{width}: max |delta keypoints| {dk:.2e} px, max |delta descriptors| {dd:.2e}")
        if dk > 1e-4 or dd > 1e-4:
            raise SystemExit("split does not reproduce the original extractor")


def export_dense(k: int, precisions: list[str]) -> None:
    import convert_coreml  # noqa: F401  (registers the coremltools patches)
    import coremltools as ct

    for height, width in SIZES:
        dense = Dense(build(k, height, width)).eval()
        image = torch.rand(1, 3, height, width)
        with torch.no_grad():
            traced = torch.jit.trace(dense, (image,), check_trace=False)
        for precision in precisions:
            model = ct.convert(
                traced, inputs=[ct.TensorType(name="image", shape=image.shape)],
                outputs=[ct.TensorType(name="logits"), ct.TensorType(name="ranker"), ct.TensorType(name="features")],
                convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
            )
            path = MODELS / f"raco_aliked_dense_{height}x{width}_{precision}.mlpackage"
            model.save(str(path))
            print(f"wrote {path}")


def export_sparse(k: int) -> None:
    from export_split import export

    for height, width in SIZES:
        extractor = build(k, height, width)
        sparse = Sparse(extractor).eval()
        image = torch.rand(1, 3, height, width)
        with torch.no_grad():
            logits, ranker, features = Dense(extractor)(image)
        export(sparse, (logits, ranker, features), ["logits", "ranker", "features"], ["keypoints", "descriptors"],
               MODELS / f"raco_aliked_sparse_k{k}_{height}x{width}.onnx")


def export_levels(k: int) -> None:
    import convert_coreml  # noqa: F401  (registers the coremltools patches)
    import coremltools as ct

    for height, width in SIZES:
        levels = Levels(build(k, height, width)).eval()
        image = torch.rand(1, 3, height, width)
        with torch.no_grad():
            traced = torch.jit.trace(levels, (image,), check_trace=False)
        model = ct.convert(
            traced, inputs=[ct.TensorType(name="image", shape=image.shape)],
            outputs=[ct.TensorType(name=name) for name in ("logits", "ranker", "level1", "level2", "level3", "level4")],
            convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT32,
        )
        path = MODELS / f"raco_aliked_levels_{height}x{width}_fp32.mlpackage"
        model.save(str(path))
        print(f"wrote {path}")


def export_select(k: int) -> None:
    from export_split import export

    for height, width in SIZES:
        extractor = build(k, height, width)
        with torch.no_grad():
            logits, ranker, _ = Dense(extractor)(torch.rand(1, 3, height, width))
        export(Select(extractor).eval(), (logits, ranker), ["logits", "ranker"], ["keypoints"],
               MODELS / f"raco_select_k{k}_{height}x{width}.onnx")


def export_head(k: int) -> None:
    """Raw little-endian file: magic, 4 int32 sizes (channels, positions, kernel, dimensions), then the
    float32 tensors offset weight, offset bias, second weight, second bias, sample weight, aggregation."""
    import numpy as np

    head = build(k, *SIZES[0]).aliked.desc_head
    first, _activation, second = head.offset_conv
    channels, positions, kernel = first.in_channels, head.positions, head.kernel_size
    dimensions = head.agg_weights.shape[2]
    tensors = [first.weight, first.bias, second.weight, second.bias, head.sf_conv.weight, head.agg_weights]
    path = MODELS / "aliked_descriptor_head.bin"
    with open(path, "wb") as file:
        file.write(b"ALKH")
        file.write(np.array([channels, positions, kernel, dimensions], dtype="<i4").tobytes())
        for tensor in tensors:
            file.write(tensor.detach().contiguous().numpy().astype("<f4").tobytes())
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "dense", "sparse", "levels", "select", "head"])
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--precision", nargs="+", default=["fp16", "fp32"])
    args = parser.parse_args()
    if args.step == "check":
        check(args.keypoints)
    elif args.step == "dense":
        export_dense(args.keypoints, args.precision)
    elif args.step == "sparse":
        export_sparse(args.keypoints)
    elif args.step == "levels":
        export_levels(args.keypoints)
    elif args.step == "select":
        export_select(args.keypoints)
    else:
        export_head(args.keypoints)


if __name__ == "__main__":
    main()
