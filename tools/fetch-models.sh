#!/bin/zsh
# Downloads the default model set listed in tools/models.json into Models/ and checks its SHA-256.
set -euo pipefail
root="${0:A:h:h}"
manifest="$root/tools/models.json"
read -r url digest archive <<<"$(/usr/bin/python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["url"], m["sha256"], m["archive"])' "$manifest")"
mkdir -p "$root/Models" "$root/build"
target="$root/build/$archive"
if [ ! -f "$target" ] || [ "$(shasum -a 256 "$target" | cut -d' ' -f1)" != "$digest" ]; then
  echo "downloading $url"
  curl -fL --progress-bar -o "$target" "$url"
fi
if [ "$(shasum -a 256 "$target" | cut -d' ' -f1)" != "$digest" ]; then
  rm -f "$target"
  echo "checksum mismatch for $archive" >&2
  exit 1
fi
unzip -qo "$target" -d "$root/Models"
echo "models ready in $root/Models"
