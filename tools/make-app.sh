#!/bin/zsh
# Builds Tessera with SwiftPM and wraps it into an ad-hoc signed .app in build/, with the default
# models from Models/ (tools/fetch-models.sh) inside the bundle.
set -euo pipefail
root="${0:A:h:h}"
configuration="${1:-release}"
cd "$root"
swift build -c "$configuration" --product Tessera
binary="$(swift build -c "$configuration" --show-bin-path)/Tessera"
app="$root/build/Tessera.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources/Models"
cp "$binary" "$app/Contents/MacOS/Tessera"
cp -R "$root/App/Resources/"*.lproj "$app/Contents/Resources/"
models=(
  raco_aliked_levels_768x1024_fp32.mlpackage raco_aliked_levels_1024x768_fp32.mlpackage
  raco_select_k2048_768x1024.onnx raco_select_k2048_1024x768.onnx
  aliked_descriptor_head.bin
  lightglue_raco_aliked_k2048_fp16.mlpackage
)
missing=0
for model in "${models[@]}"; do
  if [ -e "$root/Models/$model" ]; then
    cp -R "$root/Models/$model" "$app/Contents/Resources/Models/"
  else
    missing=1
  fi
done
if [ "$missing" = 1 ]; then
  echo "warning: default models missing from Models/ (run tools/fetch-models.sh); the learned matcher will be off" >&2
fi
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
