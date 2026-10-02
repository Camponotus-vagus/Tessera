#!/bin/zsh
# Builds the downloadable Tessera in build/release/: a release build without ONNX Runtime, ad-hoc
# signed, checked to use only system libraries and to run from macOS 15, as a zip and a DMG.
# usage: tools/make-release.sh [version]   (default: CFBundleShortVersionString from make-app.sh)
set -euo pipefail
root="${0:A:h:h}"
cd "$root"
[ -d Vendor/opencv/lib ] || tools/build-opencv.sh
TESSERA_TRAITS= tools/make-app.sh release > /dev/null
app="$root/build/Tessera.app"
version="${1:-$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")}"
out="$root/build/release"
rm -rf "$out"
mkdir -p "$out"

# Every Mach-O file in the bundle may only link system libraries and must allow macOS 15.
failed=0
while IFS= read -r file; do
  if otool -L "$file" | tail -n +2 | awk '{print $1}' | grep -vE '^(/System/Library/|/usr/lib/)' | grep -q .; then
    echo "non-system library linked by $file:" >&2
    otool -L "$file" | tail -n +2 | grep -vE '/System/Library/|/usr/lib/' >&2
    failed=1
  fi
  minos=$(otool -l "$file" | awk '/LC_BUILD_VERSION/{found=1} found && /minos/{print $2; exit}')
  if [ -n "$minos" ] && [ "${minos%%.*}" -gt 15 ]; then
    echo "$file requires macOS $minos" >&2
    failed=1
  fi
done < <(find "$app" -type f -perm -u+x -exec sh -c 'file -b "$1" | grep -q Mach-O && echo "$1"' _ {} \;)
[ "$failed" = 0 ] || exit 1
codesign --verify --deep --strict "$app"

# No extended attributes: as AppleDouble files they break the signature when unpacked with unzip.
ditto -c -k --norsrc --noextattr --noacl --keepParent "$app" "$out/Tessera-$version.zip"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
cp -R "$app" "$staging/"
ln -s /Applications "$staging/Applications"
hdiutil create -quiet -volname "Tessera $version" -srcfolder "$staging" -fs HFS+ -format UDZO "$out/Tessera-$version.dmg"
(cd "$out" && shasum -a 256 "Tessera-$version.zip" "Tessera-$version.dmg" > SHA256SUMS)
du -sh "$app" "$out"/* | sed "s|$root/||"
