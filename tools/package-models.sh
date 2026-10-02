#!/bin/zsh
# Zips the default model set from Models/ for a GitHub release and records its SHA-256 and size
# in tools/models.json (used by tools/fetch-models.sh and the app bundle script).
set -euo pipefail
root="${0:A:h:h}"
version="${1:?usage: package-models.sh <release-tag, e.g. models-1>}"
cd "$root/Models"
files=(
  raco_aliked_levels_768x1024_fp32.mlpackage raco_aliked_levels_1024x768_fp32.mlpackage
  raco_select_k2048_768x1024.onnx raco_select_k2048_1024x768.onnx
  aliked_descriptor_head.bin
  lightglue_raco_aliked_k2048_fp16.mlpackage
)
archive="$root/build/tessera-$version.zip"
mkdir -p "$root/build"
rm -f "$archive"
zip -qry "$archive" "${files[@]}"
# The licenses of the weights the models are converted from travel with them.
notices=$(mktemp -d)
cp "$root/Licenses/models.md" "$notices/LICENSES.md"
cp "$root/Licenses/Apache-2.0.txt" "$notices/"
(cd "$notices" && zip -qX "$archive" LICENSES.md Apache-2.0.txt)
rm -rf "$notices"
digest=$(shasum -a 256 "$archive" | cut -d' ' -f1)
size=$(stat -f%z "$archive")
/usr/bin/python3 - "$root/tools/models.json" "$version" "$digest" "$size" <<'PY'
import json, sys
path, version, digest, size = sys.argv[1:]
json.dump({"version": 2, "release": version, "archive": f"tessera-{version}.zip",
           "url": f"https://github.com/Camponotus-vagus/Tessera/releases/download/{version}/tessera-{version}.zip",
           "sha256": digest, "size": int(size)}, open(path, "w"), indent=2)
PY
echo "$archive"
echo "sha256 $digest, $size bytes"
