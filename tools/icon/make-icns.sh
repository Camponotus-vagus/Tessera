#!/bin/zsh
# Regenerates App/Resources/AppIcon.icns from tools/icon/make-icon.swift.
set -euo pipefail
root="${0:A:h:h:h}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
swift "$root/tools/icon/make-icon.swift" "$work/icon.png" > /dev/null
mkdir "$work/AppIcon.iconset"
for size in 16 32 128 256 512; do
  sips -z $size $size "$work/icon.png" --out "$work/AppIcon.iconset/icon_${size}x${size}.png" > /dev/null
  sips -z $((size * 2)) $((size * 2)) "$work/icon.png" --out "$work/AppIcon.iconset/icon_${size}x${size}@2x.png" > /dev/null
done
iconutil -c icns "$work/AppIcon.iconset" -o "$root/App/Resources/AppIcon.icns"
echo "$root/App/Resources/AppIcon.icns"
