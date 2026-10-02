// swift-tools-version: 6.1
import PackageDescription

// OpenCV is a minimal static build in Vendor/opencv (tools/build-opencv.sh). ONNX Runtime comes from
// Homebrew and is optional: it only runs the alternative ONNX extractors and matcher and the comparison
// tests (swift build --traits ONNXRuntime). The app does not need it.
let vendor = Context.packageDirectory + "/../../Vendor/opencv"
let brew = "/opt/homebrew/opt/onnxruntime"
let opencvLibraries = ["opencv_stitching", "opencv_features", "opencv_geometry", "opencv_flann", "opencv_imgproc",
                       "opencv_core", "tegra_hal"]
let onnx: Set<String> = ["ONNXRuntime"]

let package = Package(
    name: "StitchKit",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "StitchKit", targets: ["StitchKit"]),
        .executable(name: "stitchbench", targets: ["stitchbench"]),
    ],
    traits: [
        .trait(name: "ONNXRuntime", description: "Link ONNX Runtime for the ONNX extractors and matcher"),
    ],
    targets: [
        .target(
            name: "CStitchCore",
            cxxSettings: [
                .unsafeFlags(["-I\(vendor)/include/opencv5"]),
                // OpenCV's inline headers put __FILE__ into assertion messages: no build-machine paths.
                .unsafeFlags(["-ffile-prefix-map=\(vendor)/=Vendor/opencv/"]),
                .unsafeFlags(["-I\(brew)/include"], .when(traits: onnx)),
                .define("TESSERA_ONNXRUNTIME", .when(traits: onnx)),
            ],
            linkerSettings: [
                .unsafeFlags(["-L\(vendor)/lib", "-L\(vendor)/lib/opencv5/3rdparty"]),
                .unsafeFlags(["-L\(brew)/lib"], .when(traits: onnx)),
                .linkedLibrary("onnxruntime", .when(traits: onnx)),
            ] + opencvLibraries.map { .linkedLibrary($0) } + [.linkedLibrary("z"), .linkedFramework("Accelerate")]
        ),
        .target(
            name: "StitchKit",
            dependencies: ["CStitchCore"],
            swiftSettings: [.define("TESSERA_ONNXRUNTIME", .when(traits: onnx))]
        ),
        .executableTarget(
            name: "stitchbench",
            dependencies: ["StitchKit"]
        ),
        .testTarget(
            name: "StitchKitTests",
            dependencies: ["StitchKit", "CStitchCore"],
            swiftSettings: [.define("TESSERA_ONNXRUNTIME", .when(traits: onnx))]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
