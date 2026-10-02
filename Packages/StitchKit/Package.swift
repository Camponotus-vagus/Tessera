// swift-tools-version: 6.0
import PackageDescription

// OpenCV 5 and ONNX Runtime come from Homebrew.
let brew = "/opt/homebrew/opt"
let opencvLibraries = ["opencv_core", "opencv_imgproc", "opencv_features", "opencv_flann", "opencv_geometry"]

let package = Package(
    name: "StitchKit",
    platforms: [.macOS("27.0")],
    products: [
        .library(name: "StitchKit", targets: ["StitchKit"]),
        .executable(name: "stitchbench", targets: ["stitchbench"]),
    ],
    targets: [
        .target(
            name: "CStitchCore",
            cxxSettings: [
                .unsafeFlags(["-I\(brew)/opencv/include/opencv5", "-I\(brew)/onnxruntime/include"]),
            ],
            linkerSettings: [
                .unsafeFlags(["-L\(brew)/opencv/lib", "-L\(brew)/onnxruntime/lib"]),
            ] + opencvLibraries.map { .linkedLibrary($0) } + [.linkedLibrary("onnxruntime"), .linkedFramework("Accelerate")]
        ),
        .target(
            name: "StitchKit",
            dependencies: ["CStitchCore"]
        ),
        .executableTarget(
            name: "stitchbench",
            dependencies: ["StitchKit"]
        ),
        .testTarget(
            name: "StitchKitTests",
            dependencies: ["StitchKit", "CStitchCore"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
