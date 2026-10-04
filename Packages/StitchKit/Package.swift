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
                // SwiftPM builds C++ for release with -Os, which leaves the alignment's loops scalar. At -O2 and
                // above the loop vectoriser splits a sum of products (s += a * b) into a product and an ordered
                // sum, which changes its last bit and, through the matcher, a whole analysis: every such sum in
                // CStitchCore is written with std::fma (align.cpp, descriptor_head.cpp, geometry.cpp), which gives
                // the bits -Os gave. No fast-math.
                .unsafeFlags(["-O3"], .when(configuration: .release)),
                // Debug builds too: unoptimised, the alignment's block Cholesky ran 60 to 130 times slower.
                .unsafeFlags(["-Os"], .when(configuration: .debug)),
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
