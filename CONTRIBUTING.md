# Contributing

Bug reports with photos that break the analysis are the most useful contribution: open an issue with the photos (or a link) and the JSON report.

## Development

```bash
brew install cmake ninja
tools/build-opencv.sh
tools/fetch-models.sh
cd Packages/StitchKit && swift test
cd ../.. && tools/make-app.sh
```

For the ONNX Runtime backends, `brew install onnxruntime` and build with `--traits ONNXRuntime` (or `TESSERA_TRAITS=onnx tools/make-app.sh`).

- Swift 6 with strict concurrency; the app is SwiftUI.
- C++ stays behind the C API in `Packages/StitchKit/Sources/CStitchCore/include/stitchcore.h`; Swift owns every native handle.
- Changes to the engine need a test in `Packages/StitchKit/Tests` (synthetic tiles with known offsets work well) and, for performance work, before/after numbers from `stitchbench`.
- Model changes go through `tools/export` and must keep the equivalence checks passing: `split_extractor.py check`, and `swift test --traits ONNXRuntime`, which runs `DescriptorHeadTests` (it needs the full set from `tools/export/make-models.sh`) and the ONNX comparisons in `NativeSelectTests` (the one on real photos also needs folders of overlapping photos in `TestData/real`, or `TESSERA_TESTDATA` pointing to them). Without the trait, the models or the photos they are reported as skipped, not failed. Run them from `Packages/StitchKit` with `TESSERA_MODELS="$PWD/../../Models"`, so that a model set the app installed in Application Support is not picked first.
