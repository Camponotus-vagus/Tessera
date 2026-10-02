// swift-tools-version: 6.1
// SwiftPM build of the app, used by tools/make-app.sh. The Xcode project comes from project.yml.
import PackageDescription

let package = Package(
    name: "Tessera",
    platforms: [.macOS(.v15)],
    traits: [
        .trait(name: "ONNXRuntime", description: "Build StitchKit with ONNX Runtime (comparisons only)"),
    ],
    dependencies: [
        .package(path: "Packages/StitchKit", traits: [
            .trait(name: "ONNXRuntime", condition: .when(traits: ["ONNXRuntime"])),
        ]),
    ],
    targets: [
        .executableTarget(
            name: "Tessera",
            dependencies: [.product(name: "StitchKit", package: "StitchKit")],
            path: "App",
            exclude: ["Resources"]
        ),
    ]
)
