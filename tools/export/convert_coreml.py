"""Convert RaCo-ALIKED (extractor) and LightGlue (matcher) straight from PyTorch to Core ML packages.

ONNX Runtime's CoreML provider splits these graphs into dozens of partitions and bounces between
CPU and GPU. A native ML Program runs as one graph that Core ML can place on CPU, GPU or ANE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch import ops as torch_ops
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

from export_split import ROOT, Extract, Match  # same wrappers as the ONNX export

from lightglue_dynamo.config import Extractor, RankerMode  # noqa: E402  (path set by export_split)
from lightglue_dynamo.models import LightGlue, RaCoALIKED  # noqa: E402
from lightglue_dynamo.models.lightglue import SelfBlock  # noqa: E402
from lightglue_dynamo.ops import multi_head_attention  # noqa: E402


def _cast_numpy2(context, node, dtype, dtype_name):
    """coremltools 9.0 calls int() on 1-element arrays, which NumPy 2.5 rejects; use .item()."""
    x = torch_ops._get_inputs(context, node, expected=1)[0]
    if x.can_be_folded_to_const() and not isinstance(x.val, dtype):
        context.add(mb.const(val=dtype(np.asarray(x.val).item()), name=node.name), node.name)
        return
    _original_cast(context, node, dtype, dtype_name)


_original_cast = torch_ops._cast
torch_ops._cast = _cast_numpy2


@register_torch_op
def log_sigmoid(context, node):
    # log(sigmoid(x)) = -softplus(-x), stable for large |x|.
    x = torch_ops._get_inputs(context, node, expected=1)[0]
    softplus = mb.softplus(x=mb.mul(x=x, y=np.float32(-1)))
    context.add(mb.mul(x=softplus, y=np.float32(-1), name=node.name))


def _rotate_half(t: torch.Tensor) -> torch.Tensor:
    # Pairs (x0, x1) along the head dimension become (-x1, x0), as in LightGlue.
    pairs = t.unflatten(-1, (-1, 2))
    return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2, -1)


def _self_block_rank5(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
    """SelfBlock.forward without the rank-6 tensor Core ML rejects: rotate q and k separately."""
    b, n, _ = x.shape
    qkv = self.Wqkv(x).reshape((b, n, self.num_heads, self.head_dim, 3)).transpose(1, 2)
    q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
    cosine, sine = encoding[0], encoding[1]
    q = q * cosine + _rotate_half(q) * sine
    k = k * cosine + _rotate_half(k) * sine
    q, k, v = (tensor.transpose(1, 2).reshape(b, n, self.embed_dim) for tensor in (q, k, v))
    context = multi_head_attention(q, k, v, self.num_heads)
    return x + self.ffn(torch.concat([x, self.out_proj(context)], 2))


def use_rank5_attention(matcher: LightGlue) -> None:
    """Swap in the rank-5 self-attention after checking it matches the original numerically."""
    keypoints, descriptors = torch.rand(2, 256, 2) * 2 - 1, torch.randn(2, 256, 128)
    reference = Match(matcher)(keypoints, descriptors)
    original = SelfBlock.forward
    SelfBlock.forward = _self_block_rank5
    candidate = Match(matcher)(keypoints, descriptors)
    if not torch.equal(reference[0], candidate[0]) or (reference[1] - candidate[1]).abs().max() > 1e-5:
        SelfBlock.forward = original
        raise RuntimeError("rank-5 self-attention does not reproduce the original")
    print("rank-5 self-attention matches the original")


def convert(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], names_in: list[str], names_out: list[str],
            precision: str, path: Path) -> None:
    traced = torch.jit.trace(module, inputs, check_trace=False)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=tensor.shape) for name, tensor in zip(names_in, inputs)],
        outputs=[ct.TensorType(name=name) for name in names_out],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.save(str(path))
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--precision", choices=["fp16", "fp32"], nargs="+", default=["fp32", "fp16"])
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "Models")
    parser.add_argument("--only", choices=["extractor", "matcher"])
    args = parser.parse_args()
    k = args.keypoints

    with torch.no_grad():
        for height, width in [] if args.only == "matcher" else [(768, 1024), (1024, 768)]:
            mode = RankerMode.auto.resolve(k, height, width)
            extractor = RaCoALIKED(num_keypoints=k, portable_deform_conv=True, ranker_mode=mode).eval()
            extractor.fuse_batch_norm()
            for precision in args.precision:
                convert(Extract(extractor).eval(), (torch.rand(1, 3, height, width),), ["image"],
                        ["keypoints", "descriptors"], precision,
                        args.output / f"raco_aliked_extractor_k{k}_{height}x{width}_{precision}.mlpackage")
        matcher = LightGlue(**Extractor.raco_aliked.lightglue_config).eval()
        use_rank5_attention(matcher)
        for precision in [] if args.only == "extractor" else args.precision:
            convert(Match(matcher).eval(), (torch.rand(2, k, 2) * 2 - 1, torch.randn(2, k, 128)),
                    ["keypoints", "descriptors"], ["match0", "confidence0"], precision,
                    args.output / f"lightglue_raco_aliked_k{k}_{precision}.mlpackage")


if __name__ == "__main__":
    main()
