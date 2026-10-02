#!/bin/zsh
# Builds Tessera with SwiftPM and wraps it into an ad-hoc signed .app in build/. The learned models are not
# bundled: the app downloads them on request from the release in tools/models.json, copied in as
# Resources/models.json.
set -euo pipefail
root="${0:A:h:h}"
configuration="${1:-release}"
cd "$root"
swift build -c "$configuration" --product Tessera
binary="$(swift build -c "$configuration" --show-bin-path)/Tessera"
app="$root/build/Tessera.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$binary" "$app/Contents/MacOS/Tessera"
cp -R "$root/App/Resources/"*.lproj "$app/Contents/Resources/"
cp "$root/tools/models.json" "$app/Contents/Resources/models.json"
cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>Tessera</string>
  <key>CFBundleExecutable</key><string>Tessera</string>
  <key>CFBundleIdentifier</key><string>io.github.camponotus-vagus.Tessera</string>
  <key>CFBundleName</key><string>Tessera</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.photography</string>
  <key>LSMinimumSystemVersion</key><string>27.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
codesign --force --sign - "$app"
echo "$app"
