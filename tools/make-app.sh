#!/bin/zsh
# Builds Tessera with SwiftPM and wraps it into an ad-hoc signed .app in build/. The learned models are not
# bundled: the app downloads them on request from the release in tools/models.json, copied in as
# Resources/models.json. TESSERA_TRAITS=onnx links ONNX Runtime from Homebrew (not needed by the app).
set -euo pipefail
root="${0:A:h:h}"
configuration="${1:-release}"
version="${TESSERA_VERSION:-0.1.1}"
cd "$root"
traits=()
[ "${TESSERA_TRAITS:-}" = onnx ] && traits=(--traits ONNXRuntime)
swift build -c "$configuration" --product Tessera "${traits[@]}"
binary="$(swift build -c "$configuration" --show-bin-path "${traits[@]}")/Tessera"
app="$root/build/Tessera.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$binary" "$app/Contents/MacOS/Tessera"
# Drop the debug map and local symbols, which hold absolute paths of the build machine.
strip -S -x "$app/Contents/MacOS/Tessera"
cp -R "$root/App/Resources/"*.lproj "$app/Contents/Resources/"
cp "$root/tools/models.json" "$app/Contents/Resources/models.json"
cp "$root/App/Resources/AppIcon.icns" "$app/Contents/Resources/AppIcon.icns"
# Licenses of Tessera and of what it contains (OpenCV is linked statically).
licenses="$app/Contents/Resources/Licenses"
mkdir -p "$licenses/opencv"
cp "$root/LICENSE" "$licenses/Tessera.txt"
cp "$root/THIRD_PARTY_NOTICES.md" "$root/Licenses/Apache-2.0.txt" "$root/Licenses/opencv.md" "$licenses/"
cp "$root/Vendor/opencv/share/licenses/opencv5/"* "$licenses/opencv/"
cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>Tessera</string>
  <key>CFBundleExecutable</key><string>Tessera</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>io.github.camponotus-vagus.Tessera</string>
  <key>CFBundleName</key><string>Tessera</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>Image</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>LSItemContentTypes</key><array><string>public.image</string></array>
    </dict>
  </array>
  <key>LSApplicationCategoryType</key><string>public.app-category.photography</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>Copyright 2026 Francesco Simone Mensa. MIT License.</string>
</dict>
</plist>
PLIST
codesign --force --sign - "$app"
echo "$app"
