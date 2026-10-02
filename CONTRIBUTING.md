# Contributing

Bug reports with photos that break the analysis are the most useful contribution: open an issue with the photos (or a link) and the JSON report.

## Development

```bash
brew install opencv onnxruntime
tools/fetch-models.sh
cd Packages/StitchKit && swift test
cd ../.. && tools/make-app.sh
```

- Swift 6 with strict concurrency; the app is SwiftUI.
- C++ stays behind the C API in `Packages/StitchKit/Sources/CStitchCore/include/stitchcore.h`; Swift owns every native handle.
- Changes to the engine need a test in `Packages/StitchKit/Tests` (synthetic tiles with known offsets work well) and, for performance work, before/after numbers from `stitchbench`.
- Model changes go through `tools/export` and must keep the equivalence checks passing (`split_extractor.py check`, `DescriptorHeadTests`).
