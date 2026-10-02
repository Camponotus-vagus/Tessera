#!/bin/zsh
# Regenerates the split RaCo-ALIKED + LightGlue models in Models/ (see README.md in this folder).
set -euo pipefail
here="${0:A:h}"
cd "$here"
export TORCH_HOME="$here/weights"
# The export scripts patch and import LightGlue-ONNX modules, so they are tied to this commit.
lightglue_onnx_commit=d12b4ba
for venv in .venv .venv-coreml; do
  if [ ! -x "$venv/bin/python" ]; then
    echo "missing $here/$venv: create it as described in README.md (Environments)" >&2
    exit 1
  fi
done
if [ ! -d LightGlue-ONNX ]; then
  git clone -q https://github.com/fabio-sim/LightGlue-ONNX.git
  git -C LightGlue-ONNX checkout -q "$lightglue_onnx_commit"
fi
mkdir -p "$here/../../Models"
./.venv/bin/python split_extractor.py check
./.venv/bin/python export_split.py                     # full ONNX extractors (fallback) and ONNX matcher
./.venv/bin/python split_extractor.py sparse           # sparse extractor halves (ONNX, CPU)
./.venv-coreml/bin/python split_extractor.py dense     # dense extractor halves (Core ML)
./.venv-coreml/bin/python split_extractor.py levels    # dense maps + ALIKED feature levels (Core ML, default)
./.venv/bin/python split_extractor.py select           # keypoint selection (ONNX, default)
./.venv/bin/python split_extractor.py head             # descriptor head weights for the C++ head (default)
./.venv-coreml/bin/python convert_coreml.py --only matcher
